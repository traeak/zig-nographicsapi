//! Ring-buffered staging uploads on one queue. Sources are copied into CPU-visible staging immediately; staging and batch
//! storage never grow, and reuse waits for completed batches when needed.
//!
//! Externally synchronize the uploader and its queue. Destination and callback resource lifetimes stay caller-owned.
//! Order texture initialization and prior resource use on that queue before enqueuing uploads, including waits on other
//! queues. Flush between overlapping transfer writes.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("NoGraphicsAPI");

const UploadQueue = @This();

pub const operation_limit = 4096;
pub const alignment = 16;

pub const Options = struct {
    /// A positive multiple of 16.
    capacity: u64 = 64 * 1024 * 1024,
    /// Must be less than DeviceCaps.queue_count.
    queue_index: u32 = 0,
    /// Nonzero; normally the number of frames in flight.
    max_pending_batches: u32 = 2,
};

pub const Stats = struct {
    capacity: u64,
    bytes_in_use: u64,
    peak_bytes: u64,
    submissions: u64,
    waits: u64,
    pending_batches: u32,
    pending_operations: u32,
};

const Batch = struct {
    pool: *gpu.CommandPool,
    end: u64 = 0,
    value: u64 = 0,
};

const gpu_access: gpu.Access = .{
    .shader_read = true,
    .shader_write = true,
    .transfer_read = true,
    .transfer_write = true,
    .color_read = true,
    .color_write = true,
    .depth_stencil_read = true,
    .depth_stencil_write = true,
    .indirect_read = true,
    .index_read = true,
    .descriptor_read = true,
};

device: *gpu.Device,
heap: gpu.GpuHeap,
completion: gpu.TimelinePoint,
commands: ?*gpu.CommandBuffer = null,
/// Monotonic byte positions; the ring offset is the position modulo capacity.
head: u64 = 0,
tail: u64 = 0,
batches: []Batch,
queue_index: u32,
upload_stages: gpu.Stage = .{ .transfer = true },
upload_access: gpu.Access = .{ .transfer_read = true, .transfer_write = true },
queue_access: gpu.Access = .{ .transfer_read = true, .transfer_write = true },
texture_granularity: gpu.Uint32x3 = .{ .x = 1, .y = 1, .z = 1 },
retirement_first: u32 = 0,
retirement_count: u32 = 0,
operation_count: u32 = 0,
in_callback: bool = false,
submissions: u64 = 0,
waits: u64 = 0,
peak_bytes: u64 = 0,

pub fn init(allocator: std.mem.Allocator, device: *gpu.Device, options: Options) gpu.Error!UploadQueue {
    assert(options.capacity >= alignment and options.capacity % alignment == 0 and options.max_pending_batches != 0);
    const caps = gpu.getDeviceCaps(device);
    assert(options.queue_index < caps.queue_count);
    const heap = try gpu.createGpuHeap(device, options.capacity, .cpu_visible);
    errdefer gpu.destroyGpuHeap(heap);
    const semaphore = try gpu.createTimelineSemaphore(device, 0);
    errdefer gpu.destroyTimelineSemaphore(semaphore);
    const batches = allocator.alloc(Batch, options.max_pending_batches) catch @panic("out of memory");
    errdefer allocator.free(batches);
    for (batches, 0..) |*batch, created| {
        errdefer for (batches[0..created]) |previous| gpu.destroyCommandPool(previous.pool);
        batch.* = .{ .pool = try gpu.createCommandPool(device, options.queue_index) };
        errdefer gpu.destroyCommandPool(batch.pool);
        // Allocate the batch's command buffer up front so later uploads do not fail.
        gpu.endCommands(try gpu.beginCommands(batch.pool));
        gpu.resetCommandPool(batch.pool);
    }
    var self: UploadQueue = .{
        .device = device,
        .heap = heap,
        .completion = .{ .semaphore = semaphore, .value = 0 },
        .batches = batches,
        .queue_index = options.queue_index,
    };
    if (options.queue_index < caps.general_queue_count + caps.compute_queue_count) {
        self.upload_stages.compute = true;
        self.upload_access.shader_read = true;
        self.upload_access.shader_write = true;
        self.upload_access.descriptor_read = true;
        self.queue_access = self.upload_access;
        self.queue_access.indirect_read = true;
        if (options.queue_index < caps.general_queue_count) self.queue_access = gpu_access;
    } else {
        self.texture_granularity = caps.copy_texture_granularity;
    }
    return self;
}

/// Flushes and waits for uploads. Finish external submissions using returned timeline points first.
pub fn deinit(self: *UploadQueue, allocator: std.mem.Allocator) void {
    assert(!self.in_callback);
    self.wait();
    for (self.batches) |batch| gpu.destroyCommandPool(batch.pool);
    allocator.free(self.batches);
    gpu.destroyGpuHeap(self.heap);
    gpu.destroyTimelineSemaphore(self.completion.semaphore);
    self.* = undefined;
}

/// Retrieves timestamps and reclaims completed batches without waiting or flushing pending uploads.
pub fn reclaim(self: *UploadQueue) void {
    assert(!self.in_callback);
    if (self.retirement_count == 0) return;
    const completed = gpu.timelineCompletedValue(self.completion.semaphore);
    while (self.retirement_count != 0 and self.batches[self.retirement_first].value <= completed) {
        const batch = self.batches[self.retirement_first];
        self.tail = batch.end;
        gpu.readTimestamps(batch.pool);
        gpu.resetCommandPool(batch.pool);
        self.retirement_first = (self.retirement_first + 1) % @as(u32, @intCast(self.batches.len));
        self.retirement_count -= 1;
    }
}

fn waitOldest(self: *UploadQueue) void {
    assert(self.retirement_count != 0);
    gpu.waitTimeline(.{ .semaphore = self.completion.semaphore, .value = self.batches[self.retirement_first].value });
    self.waits += 1;
    self.reclaim();
}

fn reserve(self: *UploadQueue, byte_size: u64) gpu.GpuCpuRange(u8) {
    const capacity = self.heap.range.size;
    assert(!self.in_callback and byte_size != 0 and byte_size <= capacity);
    if (self.operation_count == operation_limit) _ = self.flush();
    while (true) {
        var offset = std.mem.alignForward(u64, self.head % capacity, alignment);
        if (byte_size > capacity - offset) offset = capacity;
        const start = self.head - self.head % capacity + offset;
        if (self.head == self.tail and self.retirement_count == 0) {
            self.head = start;
            self.tail = start;
        }
        if (start + byte_size - self.tail <= capacity) {
            self.head = start + byte_size;
            self.peak_bytes = @max(self.peak_bytes, self.head - self.tail);
            return self.heap.range.subrange(u8, start % capacity, byte_size);
        }
        const previous_count = self.retirement_count;
        self.reclaim();
        if (self.retirement_count != previous_count) continue;
        if (self.commands != null) {
            _ = self.flush();
            continue;
        }
        self.waitOldest();
    }
}

fn begin(self: *UploadQueue) *gpu.CommandBuffer {
    assert(!self.in_callback);
    if (self.commands) |commands| return commands;
    if (self.retirement_count == self.batches.len) {
        self.reclaim();
        if (self.retirement_count == self.batches.len) self.waitOldest();
    }
    const pool = self.batches[(self.retirement_first + self.retirement_count) % self.batches.len].pool;
    // The pool's command buffer was allocated at init, so beginning it again cannot fail.
    const commands = gpu.beginCommands(pool) catch unreachable;
    gpu.barrier(commands, .{ .all_commands = true }, self.queue_access, self.upload_stages, self.upload_access);
    self.commands = commands;
    return commands;
}

/// Buffer copies follow copyMemory's requirements; large sources are split to fit the staging capacity.
pub fn uploadBuffer(self: *UploadQueue, destination: gpu.GpuRange, source: []const u8) void {
    assert(!self.in_callback and source.len <= destination.size);
    var offset: u64 = 0;
    while (offset < source.len) {
        const size = @min(source.len - offset, self.heap.range.size);
        const staging = self.reserve(size);
        @memcpy(staging.cpu.?[0..size], source[offset..][0..size]);
        const target: [*]u8 = @ptrCast(destination.gpu.?);
        gpu.copyMemory(self.begin(), staging.gpuRange(), .{ .gpu = target + offset, .size = size });
        self.operation_count += 1;
        offset += size;
    }
}

/// Source covers one complete region, including pitch padding, and must fit the capacity. The region follows
/// copyMemoryToTexture's requirements.
pub fn uploadTexture(self: *UploadQueue, destination: *gpu.Texture, source: []const u8, copy: gpu.TextureCopyDesc) void {
    const staging = self.reserve(source.len);
    @memcpy(staging.cpu.?[0..source.len], source);
    gpu.copyMemoryToTexture(self.begin(), staging.gpuRange(), destination, copy);
    self.operation_count += 1;
}

/// Uploads a tightly packed region of a texture described by `description`, splitting it into pieces that fit the
/// staging capacity and the queue's texture granularity. Pitches must be zero and combined depth/stencil formats are
/// unsupported; staging must fit at least one granularity-sized block (clipped at the mip edge).
pub fn uploadTextureTiled(self: *UploadQueue, destination: *gpu.Texture, description: gpu.TextureDesc, source: []const u8, copy: gpu.TextureCopyDesc) void {
    assert(copy.mip_level < description.mip_levels and copy.row_pitch_bytes == 0 and copy.slice_pitch_bytes == 0);
    const capacity = self.heap.range.size;
    const format = gpu.getTextureFormatInfo(description.format);
    assert(format.bytes_per_block != 0 and capacity >= format.bytes_per_block and !(format.depth and format.stencil));
    const mip: u5 = @intCast(copy.mip_level);
    var region = copy;
    if (region.extent.x == 0) region.extent.x = @max(description.extent.x >> mip, 1) - region.offset.x;
    if (region.extent.y == 0) region.extent.y = @max(description.extent.y >> mip, 1) - region.offset.y;
    if (region.extent.z == 0) region.extent.z = @max(description.extent.z >> mip, 1) - region.offset.z;
    if (region.slice_count == 0) region.slice_count = description.layer_count - region.base_slice;
    const volume = description.type == .three_d;
    const slices = if (volume) region.extent.z else region.slice_count;
    const columns = std.math.divCeil(u32, region.extent.x, format.block_extent.x) catch unreachable;
    const rows = std.math.divCeil(u32, region.extent.y, format.block_extent.y) catch unreachable;
    const row_bytes = @as(u64, columns) * format.bytes_per_block;
    const slice_bytes = row_bytes * rows;
    assert(region.extent.x != 0 and region.extent.y != 0 and slices != 0 and source.len == slice_bytes * slices);
    var granularity = self.texture_granularity;
    if (!volume) granularity.z = 1;
    const tile = textureUploadTile(.{ .x = columns, .y = rows, .z = slices }, granularity, format.bytes_per_block, capacity);

    var slice: u32 = 0;
    while (slice < slices) : (slice += tile.z) {
        var part = region;
        const part_slices = @min(tile.z, slices - slice);
        part.base_slice = if (volume) 0 else region.base_slice + slice;
        part.offset.z = if (volume) region.offset.z + slice else 0;
        part.extent.z = if (volume) part_slices else 1;
        part.slice_count = if (volume) 1 else part_slices;
        var row: u32 = 0;
        while (row < rows) : (row += tile.y) {
            const part_rows = @min(tile.y, rows - row);
            part.offset.y = region.offset.y + row * format.block_extent.y;
            part.extent.y = @min(part_rows * format.block_extent.y, region.extent.y - row * format.block_extent.y);
            var column: u32 = 0;
            while (column < columns) : (column += tile.x) {
                const part_columns = @min(tile.x, columns - column);
                part.offset.x = region.offset.x + column * format.block_extent.x;
                part.extent.x = @min(part_columns * format.block_extent.x, region.extent.x - column * format.block_extent.x);
                const packed_row_bytes = @as(u64, part_columns) * format.bytes_per_block;
                const packed_slice_bytes = packed_row_bytes * part_rows;
                const staging = self.reserve(packed_slice_bytes * part_slices);
                const output = staging.cpu.?[0..staging.size];
                if (part_columns == columns and part_rows == rows) {
                    @memcpy(output, source[slice * slice_bytes ..][0..staging.size]);
                } else {
                    for (0..part_slices) |z| {
                        const input = source[(slice + z) * slice_bytes + row * row_bytes + column * format.bytes_per_block ..];
                        const slice_output = output[z * packed_slice_bytes ..][0..packed_slice_bytes];
                        if (part_columns == columns) {
                            @memcpy(slice_output, input[0..packed_slice_bytes]);
                        } else {
                            for (0..part_rows) |y|
                                @memcpy(slice_output[y * packed_row_bytes ..][0..packed_row_bytes], input[y * row_bytes ..][0..packed_row_bytes]);
                        }
                    }
                }
                gpu.copyMemoryToTexture(self.begin(), staging.gpuRange(), destination, part);
                self.operation_count += 1;
            }
        }
    }
}

/// Largest block tile within `capacity` bytes whose axes are multiples of `granularity`, or whole extents. A zero
/// granularity component means whole extents only.
pub fn textureUploadTile(blocks: gpu.Uint32x3, granularity_in: gpu.Uint32x3, bytes_per_block: u32, capacity: u64) gpu.Uint32x3 {
    var granularity = granularity_in;
    if (granularity.x == 0) granularity.x = blocks.x;
    if (granularity.y == 0) granularity.y = blocks.y;
    if (granularity.z == 0) granularity.z = blocks.z;
    const minimum: gpu.Uint32x3 = .{ .x = @min(granularity.x, blocks.x), .y = @min(granularity.y, blocks.y), .z = @min(granularity.z, blocks.z) };
    assert(capacity >= @as(u64, minimum.x) * minimum.y * minimum.z * bytes_per_block);
    var tile: gpu.Uint32x3 = .{};
    tile.x = @intCast(@min(capacity / (@as(u64, minimum.y) * minimum.z * bytes_per_block), blocks.x));
    if (tile.x < blocks.x) tile.x -= tile.x % granularity.x;
    tile.y = @intCast(@min(capacity / (@as(u64, tile.x) * minimum.z * bytes_per_block), blocks.y));
    if (tile.y < blocks.y) tile.y -= tile.y % granularity.y;
    tile.z = @intCast(@min(capacity / (@as(u64, tile.x) * tile.y * bytes_per_block), blocks.z));
    if (tile.z < blocks.z) tile.z -= tile.z % granularity.z;
    return tile;
}

/// Reserves `byte_size` (1..capacity) bytes and calls `function` with `args` followed by the command buffer and the
/// 16-byte-aligned staging range. Requires a general or compute queue. The callback fills staging and records compute
/// work, binding its pipeline/heaps and ordering its own dispatches; it must not touch the uploader or end, submit or
/// retain the command buffer. Staging stays occupied until the submission completes.
pub fn uploadWithCompute(self: *UploadQueue, byte_size: u64, comptime function: anytype, args: anytype) void {
    assert(self.upload_stages.compute);
    const staging = self.reserve(byte_size);
    const commands = self.begin();
    gpu.barrier(commands, .{ .transfer = true, .compute = true }, .{ .transfer_write = true, .shader_write = true }, .{ .compute = true }, .{
        .shader_read = true,
        .shader_write = true,
        .descriptor_read = true,
    });
    self.in_callback = true;
    @call(.auto, function, args ++ .{ commands, staging });
    gpu.barrier(commands, .{ .compute = true }, .{ .shader_write = true }, .{ .transfer = true, .compute = true }, .{
        .transfer_read = true,
        .transfer_write = true,
        .shader_read = true,
        .shader_write = true,
        .descriptor_read = true,
    });
    self.in_callback = false;
    self.operation_count += 1;
}

/// The device's per-command-buffer timestamp limit applies to each batch. Keep destinations valid until the batch is reclaimed.
pub fn writeTimestamp(self: *UploadQueue, cpu_destination: *u64) void {
    gpu.writeTimestamp(self.begin(), cpu_destination, .{ .all_commands = true });
}

/// Submits pending uploads and returns the latest completion point, borrowing this uploader's semaphore. Other queues can
/// wait on it through SubmitDesc.waits.
pub fn flush(self: *UploadQueue) gpu.TimelinePoint {
    assert(!self.in_callback);
    const commands = self.commands orelse return self.completion;
    self.reclaim();
    gpu.barrier(commands, self.upload_stages, self.upload_access, .{ .all_commands = true }, self.queue_access);
    gpu.endCommands(commands);
    self.completion.value += 1;
    gpu.submit(self.device, .{ .commands = &.{commands}, .completion = self.completion }, self.queue_index);
    const batch = &self.batches[(self.retirement_first + self.retirement_count) % self.batches.len];
    batch.end = self.head;
    batch.value = self.completion.value;
    self.retirement_count += 1;
    self.submissions += 1;
    self.commands = null;
    self.operation_count = 0;
    return self.completion;
}

/// Flushes, waits for every upload, and reclaims all batches.
pub fn wait(self: *UploadQueue) void {
    _ = self.flush();
    self.reclaim();
    if (self.retirement_count == 0) return;
    gpu.waitTimeline(self.completion);
    self.waits += 1;
    self.reclaim();
}

/// Occupied bytes and pending batches reflect retirements observed by the most recent operation.
pub fn stats(self: *const UploadQueue) Stats {
    return .{
        .capacity = self.heap.range.size,
        .bytes_in_use = self.head - self.tail,
        .peak_bytes = self.peak_bytes,
        .submissions = self.submissions,
        .waits = self.waits,
        .pending_batches = self.retirement_count,
        .pending_operations = self.operation_count,
    };
}
