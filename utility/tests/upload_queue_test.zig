const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const UploadQueue = @import("NoGraphicsAPIUtility").UploadQueue;
const testing = std.testing;

const buffer_bytes = 20 * 1024;
const readback_bytes = 128 * 1024;
pub const guard_bytes = 32;
const texture_count = 6;
const timestamp_batches = 4;
const unwritten_timestamp = 0xa5a5a5a5a5a5a5a5;
const transfer_usage: gpu.TextureUsage = .{ .transfer_source = true, .transfer_destination = true };

fn validAxis(tile: u32, size: u32, alignment: u32) bool {
    return tile != 0 and tile <= size and (tile == size or (alignment != 0 and tile % alignment == 0));
}

test "texture upload tile plans" {
    const sizes = [_]u32{ 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 37, 64, 137, 519 };
    const granularities = [_]gpu.Uint32x3{ .{ .x = 1, .y = 1, .z = 1 }, .{ .x = 16, .y = 16, .z = 8 }, .{ .x = 16, .y = 16, .z = 1 }, .{}, .{ .z = 1 } };
    const block_bytes = [_]u32{ 1, 4, 8, 16 };
    const capacities = [_]u64{ 16, 256, 272, 1024, 2048, 8192, 65536 };
    var cases: u32 = 0;
    for (sizes) |width| for (sizes) |height| for (sizes) |depth| for (granularities) |granularity| for (block_bytes) |bytes| for (capacities) |capacity| {
        const x = if (granularity.x != 0 and granularity.x < width) granularity.x else width;
        const y = if (granularity.y != 0 and granularity.y < height) granularity.y else height;
        const z = if (granularity.z != 0 and granularity.z < depth) granularity.z else depth;
        if (capacity < @as(u64, x) * y * z * bytes) continue;
        const tile = UploadQueue.textureUploadTile(.{ .x = width, .y = height, .z = depth }, granularity, bytes, capacity);
        if (!validAxis(tile.x, width, granularity.x) or !validAxis(tile.y, height, granularity.y) or !validAxis(tile.z, depth, granularity.z) or
            @as(u64, tile.x) * tile.y * tile.z * bytes > capacity)
        {
            std.debug.print("Invalid upload tile for {d} x {d} x {d} blocks at {d} bytes.\n", .{ width, height, depth, capacity });
            return error.TestUnexpectedResult;
        }
        cases += 1;
    };
    std.debug.print("Verified {d} upload tile plans, including coarse granularity and whole-mip-only transfers.\n", .{cases});
}

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

const TextureCase = struct {
    texture: u32 = 0,
    mip: u32 = 0,
    face: u32 = 0,
    bytes: u32 = 0,
    offset: u64 = 0,

    fn copy(item: TextureCase) gpu.TextureCopyDesc {
        return .{ .mip_level = item.mip, .base_slice = item.face, .slice_count = if (item.texture == 3) 1 else 0 };
    }
};

/// Shared by the compute callback test, which runs with compiled shaders.
pub const Fixture = struct {
    device: *gpu.Device,
    pool: *gpu.CommandPool,
    ring: ?UploadQueue,
    completion: gpu.TimelinePoint,
    buffer: gpu.GpuHeap,
    readback: gpu.GpuHeap,
    texture_heap: gpu.TextureHeap,
    textures: [texture_count]*gpu.Texture,
    attachments: [2]*gpu.RenderView,
    descriptions: [texture_count]gpu.TextureDesc,
    cases: [64]TextureCase,
    case_count: u32,
    expected: [readback_bytes]u8,
    failures: u32,

    pub fn init(self: *Fixture, queue_families: bool) !void {
        gpu.setErrorCallback(printError);
        self.device = gpu.createDevice(.{
            .desired_queue_count = 2,
            .desired_compute_queue_count = @intFromBool(queue_families),
            .desired_copy_queue_count = @intFromBool(queue_families),
            .timestamp_query_count = 2,
        }) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
        errdefer gpu.destroyDevice(self.device);
        self.failures = 0;
        self.ring = null;
        self.pool = try gpu.createCommandPool(self.device, 0);
        errdefer gpu.destroyCommandPool(self.pool);
        self.completion = .{ .semaphore = try gpu.createTimelineSemaphore(self.device, 0), .value = 0 };
        errdefer gpu.destroyTimelineSemaphore(self.completion.semaphore);
        self.buffer = try gpu.createGpuHeap(self.device, buffer_bytes, .gpu_only);
        errdefer gpu.destroyGpuHeap(self.buffer);
        self.readback = try gpu.createGpuHeap(self.device, readback_bytes, .readback);
        errdefer gpu.destroyGpuHeap(self.readback);
        self.descriptions = .{
            .{ .type = .three_d, .extent = .{ .x = 37, .y = 23, .z = 11 }, .format = .r8_unorm, .usage = transfer_usage },
            .{ .type = .two_d_array, .extent = .{ .x = 519, .y = 7, .z = 1 }, .layer_count = 5, .format = .r8_unorm, .usage = transfer_usage },
            .{ .extent = .{ .x = 137, .y = 9, .z = 1 }, .mip_levels = 8, .format = .bc6h_ufloat, .usage = transfer_usage },
            .{ .type = .cube, .extent = .{ .x = 9, .y = 9, .z = 1 }, .mip_levels = 4, .layer_count = 6, .format = .bc6h_ufloat, .usage = transfer_usage },
            .{
                .extent = .{ .x = 7, .y = 5, .z = 1 },
                .format = .rgba8_unorm,
                .usage = .{ .color_attachment = true, .transfer_source = true, .transfer_destination = true },
            },
            .{
                .extent = .{ .x = 7, .y = 5, .z = 1 },
                .format = .d32_float,
                .usage = .{ .depth_stencil_attachment = true, .transfer_source = true, .transfer_destination = true },
            },
        };
        var heap_bytes: u64 = 0;
        var offsets: [texture_count]u64 = undefined;
        for (&self.descriptions, &offsets, 0..) |*description, *offset, index| {
            if ((index == 2 or index == 3) and !gpu.getDeviceCaps(self.device).texture_compression_bc) description.format = .rgba8_unorm;
            if (!gpu.supportsTextureFormat(self.device, description.format, description.usage)) return error.SkipZigTest;
            const size = try gpu.getTextureSizeAlign(self.device, description.*);
            offset.* = std.mem.alignForward(u64, heap_bytes, size.alignment);
            heap_bytes = offset.* + size.size;
        }
        self.texture_heap = try gpu.createTextureHeap(self.device, heap_bytes);
        errdefer gpu.destroyTextureHeap(self.texture_heap);
        const initialization = try gpu.beginCommands(self.pool);
        for (&self.textures, 0..) |*texture, created| {
            errdefer for (self.textures[0..created]) |previous| gpu.destroyTexture(previous);
            texture.* = try gpu.createTexture(initialization, self.descriptions[created], self.texture_heap, offsets[created]);
        }
        errdefer for (self.textures) |texture| gpu.destroyTexture(texture);
        gpu.endCommands(initialization);
        self.submitAndWait(initialization, &.{});
        for (&self.attachments, 0..) |*attachment, created| {
            errdefer for (self.attachments[0..created]) |previous| gpu.destroyRenderView(previous);
            attachment.* = try gpu.createRenderView(self.textures[created + 4], .{});
        }
        errdefer for (self.attachments) |attachment| gpu.destroyRenderView(attachment);
        self.ring = try UploadQueue.init(testing.allocator, self.device, .{ .capacity = 256 });

        @memset(&self.expected, 0xa5);
        self.case_count = 0;
        var output: u64 = buffer_bytes + guard_bytes * 2;
        for (self.descriptions, 0..) |description, index| {
            const format = gpu.getTextureFormatInfo(description.format);
            for (0..if (description.type == .cube) 6 else 1) |face| for (0..description.mip_levels) |mip| {
                const level: u5 = @intCast(mip);
                const width = @max(description.extent.x >> level, 1);
                const height = @max(description.extent.y >> level, 1);
                const depth = @max(description.extent.z >> level, 1);
                var bytes = (std.math.divCeil(u32, width, format.block_extent.x) catch unreachable) *
                    (std.math.divCeil(u32, height, format.block_extent.y) catch unreachable) * format.bytes_per_block * depth;
                if (description.type == .two_d_array) bytes *= description.layer_count;
                output = std.mem.alignForward(u64, output, 16);
                self.cases[self.case_count] = .{ .texture = @intCast(index), .mip = @intCast(mip), .face = @intCast(face), .bytes = bytes, .offset = output };
                self.case_count += 1;
                for (self.expected[output..][0..bytes], 0..) |*byte, offset|
                    byte.* = @truncate(offset * 37 + (offset >> 7) * 19 + index * 23 + face * 53 + mip * 11);
                if (index == 5) for (0..bytes / @sizeOf(f32)) |pixel| {
                    const value = @as(f32, @floatFromInt(pixel + 1)) / 64.0;
                    @memcpy(self.expected[output + pixel * @sizeOf(f32) ..][0..@sizeOf(f32)], std.mem.asBytes(&value));
                };
                output += bytes + guard_bytes;
            };
        }
        self.check(output < readback_bytes, "fixture fits its guarded readback allocation");
    }

    pub fn deinit(self: *Fixture) void {
        gpu.waitIdle(self.device);
        if (self.ring) |*ring| ring.deinit(testing.allocator);
        gpu.destroyCommandPool(self.pool);
        for (self.attachments) |attachment| gpu.destroyRenderView(attachment);
        for (self.textures) |texture| gpu.destroyTexture(texture);
        gpu.destroyTextureHeap(self.texture_heap);
        gpu.destroyGpuHeap(self.buffer);
        gpu.destroyGpuHeap(self.readback);
        gpu.destroyTimelineSemaphore(self.completion.semaphore);
        gpu.destroyDevice(self.device);
    }

    pub fn check(self: *Fixture, condition: bool, message: []const u8) void {
        if (condition) return;
        std.debug.print("FAIL: {s}\n", .{message});
        self.failures += 1;
    }

    pub fn replaceRing(self: *Fixture, options: UploadQueue.Options) !void {
        if (self.ring) |*ring| ring.deinit(testing.allocator);
        self.ring = null;
        self.ring = try UploadQueue.init(testing.allocator, self.device, options);
    }

    fn submitAndWait(self: *Fixture, commands: *gpu.CommandBuffer, waits: []const gpu.TimelinePoint) void {
        self.completion.value += 1;
        gpu.submit(self.device, .{ .commands = &.{commands}, .waits = waits, .completion = self.completion }, 0);
        gpu.waitTimeline(self.completion);
        gpu.resetCommandPool(self.pool);
    }

    pub fn readResults(self: *Fixture) !void {
        var uploaded: [1]gpu.TimelinePoint = undefined;
        var wait_count: usize = 0;
        if (self.ring) |*ring| {
            uploaded[0] = ring.flush();
            ring.wait();
            wait_count = 1;
        }
        const readback = self.readback.range.cpu.?[0..readback_bytes];
        @memset(readback, 0xa5);
        const commands = try gpu.beginCommands(self.pool);
        gpu.copyMemory(commands, self.buffer.gpuRange(), self.readback.range.subrange(u8, guard_bytes, buffer_bytes).gpuRange());
        for (self.cases[0..self.case_count]) |item|
            gpu.copyTextureToMemory(commands, self.textures[item.texture], self.readback.range.subrange(u8, item.offset, item.bytes).gpuRange(), item.copy());
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(commands);
        self.submitAndWait(commands, uploaded[0..wait_count]);
        self.check(std.mem.eql(u8, &self.expected, readback), "all buffer/texture bytes and readback sentinels match");
    }

    pub fn runSmallRing(self: *Fixture) !void {
        const ring = &self.ring.?;
        ring.uploadBuffer(self.buffer.gpuRange(), self.expected[guard_bytes..][0..buffer_bytes]);
        _ = ring.flush();
        var bytes: [4096]u8 = @splat(0);
        for (0..64) |iteration| {
            for (bytes[0..128], 0..) |*byte, index| byte.* = @truncate(iteration * 31 + index * 7);
            @memcpy(self.expected[guard_bytes + 4096 + iteration * 128 ..][0..128], bytes[0..128]);
            ring.uploadBuffer(self.buffer.range.subrange(u8, 4096 + iteration * 128, 128).gpuRange(), bytes[0..128]);
        }
        for (&bytes, 0..) |*byte, index| byte.* = @truncate(index * 43 + (index >> 8) * 17);
        @memcpy(self.expected[guard_bytes + 512 ..][0..256], bytes[0..256]);
        ring.uploadBuffer(self.buffer.range.subrange(u8, 512, 256).gpuRange(), bytes[0..256]);
        @memcpy(self.expected[guard_bytes + 12288 ..][0..bytes.len], &bytes);
        ring.uploadBuffer(self.buffer.range.subrange(u8, 12288, bytes.len).gpuRange(), &bytes);
        _ = ring.flush();
        for (0..64) |iteration| {
            const value = 0x91a60f23 *% @as(u32, @intCast(iteration + 1));
            @memcpy(self.expected[guard_bytes + 256 + iteration * 4 ..][0..4], std.mem.asBytes(&value));
            ring.uploadBuffer(self.buffer.range.subrange(u8, 256 + iteration * 4, 4).gpuRange(), std.mem.asBytes(&value));
            _ = ring.flush();
            self.check(ring.stats().pending_batches <= 2, "default batch limit bounds pending batches across repeated submissions");
        }
        for (self.cases[0..self.case_count]) |item|
            ring.uploadTextureTiled(self.textures[item.texture], self.descriptions[item.texture], self.expected[item.offset..][0..item.bytes], item.copy());
        _ = ring.flush();
        for (bytes[0..252], 0..) |*byte, index| byte.* = @truncate(173 + index * 13);
        const region: gpu.TextureCopyDesc = .{ .offset = .{ .x = 3, .y = 5, .z = 2 }, .extent = .{ .x = 7, .y = 9, .z = 4 } };
        ring.uploadTextureTiled(self.textures[0], self.descriptions[0], bytes[0..252], region);
        for (0..4) |z| for (0..9) |y| @memcpy(self.expected[self.cases[0].offset + ((z + 2) * 23 + y + 5) * 37 + 3 ..][0..7], bytes[(z * 9 + y) * 7 ..][0..7]);
        ring.uploadTextureTiled(self.textures[1], self.descriptions[1], bytes[0..24], .{
            .base_slice = 2,
            .slice_count = 2,
            .offset = .{ .x = 5, .y = 1 },
            .extent = .{ .x = 3, .y = 4, .z = 1 },
        });
        for (0..2) |z| for (0..4) |y| @memcpy(self.expected[self.cases[1].offset + ((z + 2) * 7 + y + 1) * 519 + 5 ..][0..3], bytes[(z * 4 + y) * 3 ..][0..3]);
        try self.readResults();
        const stats = ring.stats();
        self.check(stats.capacity == 256 and stats.peak_bytes <= 256, "oversized uploads and repeated wraps retain the fixed capacity");
        self.check(stats.pending_operations == 0 and stats.pending_batches == 0 and
            stats.bytes_in_use == 0, "wait drains commands and retires every staged byte");
        std.debug.print("Upload queue: {d} submissions, {d} waits, peak {d} bytes; buffer and texture regions verified.\n", .{
            stats.submissions,
            stats.waits,
            stats.peak_bytes,
        });
    }

    fn runOperationLimit(self: *Fixture) !void {
        try self.replaceRing(.{ .capacity = 128 * 1024 });
        const ring = &self.ring.?;
        const submissions = ring.stats().submissions;
        for (0..UploadQueue.operation_limit + 1) |index| {
            const value = 0xde817f03 *% @as(u32, @intCast(index + 1));
            @memcpy(self.expected[guard_bytes + 32 + index * 4 ..][0..4], std.mem.asBytes(&value));
            ring.uploadBuffer(self.buffer.range.subrange(u8, 32 + index * 4, 4).gpuRange(), std.mem.asBytes(&value));
        }
        self.check(ring.stats().submissions == submissions + 1 and
            ring.stats().pending_operations == 1, "copy count flushes before staging exceeds the operation limit");
        try self.readResults();
        self.check(ring.stats().capacity == 128 * 1024, "operation pressure does not grow the ring");
    }

    fn runReclaimBeforeFlush(self: *Fixture) !void {
        try self.replaceRing(.{ .capacity = 256 });
        const ring = &self.ring.?;
        var bytes: [320]u8 = undefined;
        for (&bytes, 0..) |*byte, index| byte.* = @truncate(index * 53 + 17);
        @memcpy(self.expected[guard_bytes..][0..bytes.len], &bytes);
        ring.uploadBuffer(self.buffer.range.subrange(u8, 0, 128).gpuRange(), bytes[0..128]);
        _ = ring.flush();
        gpu.waitIdle(self.device);
        ring.uploadBuffer(self.buffer.range.subrange(u8, 128, 64).gpuRange(), bytes[128..192]);
        self.check(ring.stats().bytes_in_use == 192 and ring.stats().pending_batches == 1, "completed retirement remains uncollected before pressure");
        const submissions = ring.stats().submissions;
        ring.uploadBuffer(self.buffer.range.subrange(u8, 192, 128).gpuRange(), bytes[192..320]);
        self.check(ring.stats().submissions == submissions and
            ring.stats().pending_operations == 2, "completed staging space is reclaimed without flushing pending copies");
        try self.readResults();
    }

    fn runNonPowerOfTwo(self: *Fixture, max_pending_batches: u32) !void {
        try self.replaceRing(.{ .capacity = 272, .max_pending_batches = max_pending_batches });
        const ring = &self.ring.?;
        var bytes: [272]u8 = undefined;
        var offset: u64 = 1024;
        for (0..8) |iteration| for ([_]u32{ 272, 4, 20 }) |size| {
            for (bytes[0..size], 0..) |*byte, index| byte.* = @truncate(iteration * 71 + size * 3 + index * 19);
            @memcpy(self.expected[guard_bytes + offset ..][0..size], bytes[0..size]);
            ring.uploadBuffer(self.buffer.range.subrange(u8, offset, size).gpuRange(), bytes[0..size]);
            offset += size;
        };
        try self.readResults();
        const stats = ring.stats();
        self.check(stats.capacity == 272 and stats.peak_bytes == 272 and
            stats.bytes_in_use == 0, "unaligned tails and exact-capacity uploads wrap within a fixed non-power-of-two ring");
    }

    fn runEmptyBatchesAndDestroy(self: *Fixture) !void {
        try self.replaceRing(.{ .capacity = 256 });
        const ring = &self.ring.?;
        var storage: [timestamp_batches * 2 + guard_bytes * 2 / @sizeOf(u64)]u64 = @splat(unwritten_timestamp);
        const timestamps = storage[guard_bytes / @sizeOf(u64) ..][0 .. timestamp_batches * 2];
        var completion: gpu.TimelinePoint = undefined;
        for (0..timestamp_batches) |index| {
            ring.writeTimestamp(&timestamps[index * 2]);
            ring.writeTimestamp(&timestamps[index * 2 + 1]);
            completion = ring.flush();
        }
        gpu.waitTimeline(completion);
        self.check(
            timestamps[(timestamp_batches - 1) * 2] == unwritten_timestamp,
            "submission completion leaves CPU timestamp destinations unchanged until retrieval",
        );
        const waits = ring.stats().waits;
        ring.reclaim();
        self.check(ring.stats().waits == waits, "completed timestamp batches are reclaimed without another wait");
        const stats = ring.stats();
        self.check(stats.submissions == timestamp_batches and stats.bytes_in_use == 0 and
            stats.pending_batches == 0, "timestamp-only batches submit and retire without reserving staging bytes");
        for (0..timestamp_batches) |index| {
            const first = timestamps[index * 2];
            const second = timestamps[index * 2 + 1];
            self.check(first != unwritten_timestamp and second != unwritten_timestamp and
                first <= second, "timestamp-only batches retrieve both ordered timestamps");
        }
        const storage_bytes = std.mem.sliceAsBytes(&storage);
        for (0..guard_bytes) |index|
            self.check(storage_bytes[index] == 0xa5 and storage_bytes[storage_bytes.len - guard_bytes + index] == 0xa5, "timestamp guards remain unchanged");
        var destruction_timestamps: [2]u64 = @splat(unwritten_timestamp);
        var bytes: [64]u8 = undefined;
        for (&bytes, 0..) |*byte, index| byte.* = @truncate(index * 29 + 67);
        @memcpy(self.expected[guard_bytes..][0..bytes.len], &bytes);
        ring.writeTimestamp(&destruction_timestamps[0]);
        ring.uploadBuffer(self.buffer.range.subrange(u8, 0, bytes.len).gpuRange(), &bytes);
        ring.writeTimestamp(&destruction_timestamps[1]);
        self.check(ring.stats().pending_operations == 1, "destruction starts with an unsubmitted upload");
        ring.deinit(testing.allocator);
        self.ring = null;
        self.check(destruction_timestamps[0] != unwritten_timestamp and destruction_timestamps[1] != unwritten_timestamp and
            destruction_timestamps[0] <= destruction_timestamps[1], "destruction retrieves timestamps before destroying command pools");
        try self.readResults();
    }

    fn runDestroyPendingCopies(self: *Fixture) !void {
        var bytes: [64]u8 = undefined;
        for (&bytes, 0..) |*byte, index| byte.* = @truncate(index * 41 + 93);
        @memcpy(self.expected[guard_bytes + 512 ..][0..bytes.len], &bytes);
        var queue = try UploadQueue.init(testing.allocator, self.device, .{ .capacity = 256, .max_pending_batches = 3 });
        queue.uploadBuffer(self.buffer.range.subrange(u8, 512, bytes.len).gpuRange(), &bytes);
        self.check(queue.stats().pending_operations == 1, "a separate uploader holds unsubmitted copies");
        queue.deinit(testing.allocator);
        try self.readResults();
    }

    fn runPitchedTexture(self: *Fixture) !void {
        var bytes: [160]u8 = @splat(0xd7);
        for (0..5) |row| for (0..28) |column| {
            bytes[row * 32 + column] = @truncate(row * 53 + column * 19);
        };
        for (self.cases[0..self.case_count]) |item| if (item.texture == 4) for (0..5) |row| {
            @memcpy(self.expected[item.offset + row * 28 ..][0..28], bytes[row * 32 ..][0..28]);
        };
        self.ring.?.uploadTexture(self.textures[4], &bytes, .{ .row_pitch_bytes = 32, .slice_pitch_bytes = 160 });
        try self.readResults();
    }

    fn runAttachmentConsumers(self: *Fixture) !void {
        const ring = &self.ring.?;
        for (self.cases[0..self.case_count]) |item| {
            if (item.texture < 4) continue;
            const expected = self.expected[item.offset..][0..item.bytes];
            if (item.texture == 4) {
                for (expected) |*byte| byte.* ^= 0x5a;
            } else for (0..item.bytes / @sizeOf(f32)) |pixel| {
                const depth = @as(f32, @floatFromInt(64 - pixel)) / 128.0;
                @memcpy(expected[pixel * @sizeOf(f32) ..][0..@sizeOf(f32)], std.mem.asBytes(&depth));
            }
            ring.uploadTextureTiled(self.textures[item.texture], self.descriptions[item.texture], expected, .{});
        }
        _ = ring.flush();
        const commands = try gpu.beginCommands(self.pool);
        gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = self.attachments[0] }}, .depth = .{ .render_view = self.attachments[1] } }, .{});
        gpu.endRenderPass(commands);
        const attachment_stages: gpu.Stage = .{ .color_output = true, .depth_stencil_tests = true };
        gpu.barrier(commands, attachment_stages, .{ .color_write = true, .depth_stencil_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        gpu.endCommands(commands);
        self.submitAndWait(commands, &.{});
        try self.readResults();
    }

    pub fn orderQueue(self: *Fixture, queue_index: u32) !void {
        const pool = try gpu.createCommandPool(self.device, queue_index);
        defer gpu.destroyCommandPool(pool);
        const commands = try gpu.beginCommands(pool);
        gpu.endCommands(commands);
        const previous = self.completion;
        self.completion.value += 1;
        gpu.submit(self.device, .{ .commands = &.{commands}, .waits = &.{previous}, .completion = self.completion }, queue_index);
        gpu.waitTimeline(self.completion);
    }

    fn runMultipleQueues(self: *Fixture, queue_index: u32) !void {
        if (gpu.getDeviceCaps(self.device).queue_count < 2) {
            std.debug.print("Cross-queue upload checks skipped: one compatible queue.\n", .{});
            return;
        }
        try self.replaceRing(.{ .capacity = 256, .queue_index = queue_index, .max_pending_batches = 3 });
        const ring = &self.ring.?;
        try self.orderQueue(queue_index);
        var pending = ring.flush();
        pending.value += 1;
        var values: [64]u32 = undefined;
        for (&values, 0..) |*value, index| value.* = @as(u32, @intCast(index)) *% 0x91a60723 + 11;
        const value_bytes = std.mem.sliceAsBytes(&values);
        const destination = self.buffer.range.subrange(u8, 0, value_bytes.len).gpuRange();
        ring.uploadBuffer(destination, value_bytes);
        self.check(ring.stats().pending_operations == 1, "a nonzero-queue uploader holds pending copies");
        @memcpy(self.expected[guard_bytes..][0..value_bytes.len], value_bytes);
        const readback = self.readback.range.cpu.?[0 .. value_bytes.len + guard_bytes * 2];
        @memset(readback, 0xa5);
        const consumer = try gpu.beginCommands(self.pool);
        gpu.copyMemory(consumer, destination, self.readback.range.subrange(u8, guard_bytes, value_bytes.len).gpuRange());
        gpu.barrier(consumer, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(consumer);
        // Queue zero waits first, so the uploader's nonzero queue must make progress on its own.
        self.completion.value += 1;
        gpu.submit(self.device, .{ .commands = &.{consumer}, .waits = &.{pending}, .completion = self.completion }, 0);
        const uploaded = ring.flush();
        const empty_flush = ring.flush();
        self.check(uploaded.semaphore == pending.semaphore and uploaded.value == pending.value and empty_flush.semaphore == uploaded.semaphore and
            empty_flush.value == uploaded.value, "flush returns the latest completion even without a pending batch");
        gpu.waitTimeline(self.completion);
        gpu.resetCommandPool(self.pool);
        self.check(std.mem.eql(u8, value_bytes, readback[guard_bytes..][0..value_bytes.len]), "upload completion supplies the cross-queue memory dependency");
        for (0..guard_bytes) |index|
            self.check(readback[index] == 0xa5 and readback[guard_bytes + value_bytes.len + index] == 0xa5, "cross-queue readback guards remain unchanged");
        ring.wait();
    }

    fn runMaxPendingBatches(self: *Fixture, max_pending_batches: u32) !void {
        if (gpu.getDeviceCaps(self.device).queue_count < 2) return;
        try self.replaceRing(.{ .capacity = 1024, .max_pending_batches = max_pending_batches });
        const ring = &self.ring.?;
        const gate = try gpu.createTimelineSemaphore(self.device, 0);
        defer gpu.destroyTimelineSemaphore(gate);
        const release_pool = try gpu.createCommandPool(self.device, 1);
        defer gpu.destroyCommandPool(release_pool);
        const blocked = try gpu.beginCommands(self.pool);
        gpu.endCommands(blocked);
        self.completion.value += 1;
        gpu.submit(self.device, .{ .commands = &.{blocked}, .waits = &.{.{ .semaphore = gate, .value = 1 }}, .completion = self.completion }, 0);
        for (0..max_pending_batches) |index| {
            const value = 0xcafe0000 + @as(u32, @intCast(index));
            ring.uploadBuffer(self.buffer.range.subrange(u8, index * 4, 4).gpuRange(), std.mem.asBytes(&value));
            _ = ring.flush();
            @memcpy(self.expected[guard_bytes + index * 4 ..][0..4], std.mem.asBytes(&value));
        }
        self.check(ring.stats().pending_batches == max_pending_batches and
            ring.stats().waits == 0, "configured number of batches remains in flight without waiting while the queue is gated");
        const release = try gpu.beginCommands(release_pool);
        gpu.endCommands(release);
        gpu.submit(self.device, .{ .commands = &.{release}, .completion = .{ .semaphore = gate, .value = 1 } }, 1);
        const final_value: u32 = 0x1973abcd;
        ring.uploadBuffer(self.buffer.range.subrange(u8, 256, 4).gpuRange(), std.mem.asBytes(&final_value));
        self.check(ring.stats().pending_batches < max_pending_batches and ring.stats().pending_operations == 1, "an additional batch reuses a completed slot");
        @memcpy(self.expected[guard_bytes + 256 ..][0..4], std.mem.asBytes(&final_value));
        ring.wait();
        gpu.waitTimeline(self.completion);
        gpu.waitTimeline(.{ .semaphore = gate, .value = 1 });
        gpu.resetCommandPool(self.pool);
        try self.readResults();
    }

    fn runQueueTextures(self: *Fixture, queue_index: u32) !void {
        try self.replaceRing(.{ .capacity = 8192, .queue_index = queue_index });
        const ring = &self.ring.?;
        try self.orderQueue(queue_index);
        for (self.cases[0..self.case_count]) |item| {
            if (item.texture == 5) continue; // Depth copies require a general queue.
            const expected = self.expected[item.offset..][0..item.bytes];
            for (expected) |*byte| byte.* ^= @truncate(queue_index * 17 + 53);
            ring.uploadTextureTiled(self.textures[item.texture], self.descriptions[item.texture], expected, item.copy());
        }
        try self.readResults();
        self.check(ring.stats().peak_bytes <= 8192, "dedicated queue texture uploads retain bounded staging storage");
    }
};

fn run(queue_families: bool) !void {
    var fixture: Fixture = undefined;
    try fixture.init(queue_families);
    defer fixture.deinit();
    try fixture.runSmallRing();
    try fixture.runPitchedTexture();
    try fixture.runAttachmentConsumers();
    try fixture.runReclaimBeforeFlush();
    try fixture.runEmptyBatchesAndDestroy();
    try fixture.runDestroyPendingCopies();
    try fixture.runOperationLimit();
    for ([_]u32{ 1, 2, 3 }) |max_pending_batches| {
        try fixture.runNonPowerOfTwo(max_pending_batches);
        try fixture.runMaxPendingBatches(max_pending_batches);
    }
    try fixture.runMultipleQueues(1);
    if (queue_families) {
        const caps = gpu.getDeviceCaps(fixture.device);
        try fixture.runMultipleQueues(caps.general_queue_count);
        try fixture.runQueueTextures(caps.general_queue_count);
        try fixture.runMultipleQueues(caps.general_queue_count + caps.compute_queue_count);
        try fixture.runQueueTextures(caps.general_queue_count + caps.compute_queue_count);
    }
    std.debug.print("Upload queue: {d} failures.\n", .{fixture.failures});
    try testing.expectEqual(0, fixture.failures);
}

test "upload queue" {
    try run(false);
}

test "upload queue with dedicated compute and copy queues" {
    try run(true);
}
