const std = @import("std");
const zla = @import("zla");
const Io = std.Io;

const T = f64;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const io = init.io;

    std.debug.print("zla benchmark (element type: {s})\n", .{@typeName(T)});
    std.debug.print("Square matrix suite covers all public APIs.\n", .{});
    std.debug.print("Command: zig build bench\n\n", .{});

    try benchSquare(2, allocator, io);
    try benchSquare(4, allocator, io);
    try benchSquare(8, allocator, io);
    try benchSquare(16, allocator, io);
    try benchSquare(24, allocator, io);

    std.debug.print("\nVector-size suites (rectangular mat-vec scenarios)\n", .{});
    try benchVectorCase(8, 4, allocator, io);
    try benchVectorCase(16, 8, allocator, io);
    try benchVectorCase(24, 12, allocator, io);
    try benchVectorCase(32, 16, allocator, io);
}

fn benchSquare(comptime n: usize, allocator: std.mem.Allocator, io: Io) !void {
    const MatN = zla.Mat(T, n, n);

    const a_data_base = makeGeneralSquareData(n, 1);
    const b_data_base = makeGeneralSquareData(n, 2);
    const spd_data_base = makeSpdSquareData(n);
    const rhs_base = makeVectorData(n, 3);

    const a = MatN.init(a_data_base);
    const b = MatN.init(b_data_base);
    const spd = MatN.init(spd_data_base);

    std.debug.print("[square n={d}]\n", .{n});

    {
        const iters = scaledIters(80_000_000, n * n, 200, 200_000);
        var sink: T = 0;
        var data = a_data_base;
        var state: u64 = 0x9e37_79b9_7f4a_7c15 +% n;

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            data[0] = a_data_base[0] + nextPerturb(&state);
            const m = MatN.init(data);
            sink += m.data[0][0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("init", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(24_000_000, n * n, 40, 20_000);
        var sink: usize = 0;
        var matrix = a;
        var state: u64 = 0xd1b5_4a32_d192_ed03 +% n;
        var buf = try std.ArrayList(u8).initCapacity(allocator, n * n * 16 + n * 2 + 64);
        defer buf.deinit(allocator);

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            matrix.data[0][0] = a.data[0][0] + nextPerturb(&state);
            buf.clearRetainingCapacity();

            var writer = std.Io.Writer.fromArrayList(&buf);
            try writer.print("{f}", .{matrix});
            buf = writer.toArrayList();

            sink +%= buf.items.len;
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("format", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(60_000_000, n * n * n, 20, 30_000);
        var sink: T = 0;
        var b_dyn = b;
        var state: u64 = 0x94d0_49bb_1331_11eb +% n;
        var c = MatN.init(zeroData(n, n));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            b_dyn.data[0][0] = b.data[0][0] + nextPerturb(&state);
            a.mat_mul(&b_dyn, &c);
            sink += c.data[0][0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("mat_mul", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(80_000_000, n * n, 100, 200_000);
        var sink: T = 0;
        var rhs = rhs_base;
        var state: u64 = 0x1234_5678_9abc_def0 +% n;
        var out = @as(@Vector(n, T), @splat(0));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            rhs[0] = rhs_base[0] + nextPerturb(&state);
            a.vec_mul(&rhs, &out);
            sink += out[0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("vec_mul", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(18_000_000, n * n * n, 4, 4_000);
        var sink: T = 0;
        var matrix = a;
        var state: u64 = 0x243f_6a88_85a3_08d3 +% n;
        var inv = MatN.init(zeroData(n, n));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            matrix.data[0][0] = a.data[0][0] + nextPerturb(&state);
            try matrix.mat_inv(&inv);
            sink += inv.data[0][0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("mat_inv", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(30_000_000, n * n * n, 8, 8_000);
        var sink: T = 0;
        var rhs = rhs_base;
        var state: u64 = 0xa409_3822_299f_31d0 +% n;
        var x = @as(@Vector(n, T), @splat(0));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            rhs[0] = rhs_base[0] + nextPerturb(&state);
            try a.solve_lu(&rhs, &x);
            sink += x[0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("solve_lu", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(30_000_000, n * n * n, 8, 8_000);
        var sink: T = 0;
        var rhs = rhs_base;
        var state: u64 = 0x082e_fa98_ec4e_6c89 +% n;
        var x = @as(@Vector(n, T), @splat(0));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            rhs[0] = rhs_base[0] + nextPerturb(&state);
            try spd.solve_cholesky(&rhs, &x);
            sink += x[0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("solve_cholesky", iters, elapsedNs(io, start_ns));
    }

    std.debug.print("\n", .{});
}

fn benchVectorCase(comptime rows: usize, comptime cols: usize, allocator: std.mem.Allocator, io: Io) !void {
    const MatRC = zla.Mat(T, rows, cols);
    const MatC1 = zla.Mat(T, cols, 1);
    const MatR1 = zla.Mat(T, rows, 1);

    const a_data_base = makeRectData(rows, cols, 7);
    const vec_base = makeVectorData(cols, 11);

    const a = MatRC.init(a_data_base);

    std.debug.print("[vector rows={d}, cols={d}]\n", .{ rows, cols });

    {
        const iters = scaledIters(60_000_000, rows * cols, 200, 200_000);
        var sink: T = 0;
        var data = a_data_base;
        var state: u64 = 0x4528_21e6_38d0_1377 +% rows +% cols;

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            data[0] = a_data_base[0] + nextPerturb(&state);
            const m = MatRC.init(data);
            sink += m.data[0][0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("init", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(20_000_000, rows * cols, 40, 20_000);
        var sink: usize = 0;
        var matrix = a;
        var state: u64 = 0xbe54_66cf_34e9_0c6c +% rows +% cols;
        var buf = try std.ArrayList(u8).initCapacity(allocator, rows * cols * 16 + rows * 2 + 64);
        defer buf.deinit(allocator);

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            matrix.data[0][0] = a.data[0][0] + nextPerturb(&state);
            buf.clearRetainingCapacity();

            var writer = std.Io.Writer.fromArrayList(&buf);
            try writer.print("{f}", .{matrix});
            buf = writer.toArrayList();

            sink +%= buf.items.len;
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("format", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(80_000_000, rows * cols, 100, 200_000);
        var sink: T = 0;
        var vec = vec_base;
        var state: u64 = 0xc0ac_29b7_c97c_50dd +% rows +% cols;
        var out = @as(@Vector(rows, T), @splat(0));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            vec[0] = vec_base[0] + nextPerturb(&state);
            a.vec_mul(&vec, &out);
            sink += out[0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("vec_mul", iters, elapsedNs(io, start_ns));
    }

    {
        const iters = scaledIters(70_000_000, rows * cols, 100, 200_000);
        var sink: T = 0;
        var state: u64 = 0x3f84_d5b5_b547_0917 +% rows +% cols;
        var rhs_matrix = MatC1.init(@as([cols]T, vec_base));
        var out_matrix = MatR1.init(zeroData(rows, 1));

        var i: usize = 0;
        const start_ns = nowNs(io);
        while (i < iters) : (i += 1) {
            rhs_matrix.data[0][0] = vec_base[0] + nextPerturb(&state);
            a.mat_mul(&rhs_matrix, &out_matrix);
            sink += out_matrix.data[0][0];
        }

        std.mem.doNotOptimizeAway(sink);
        std.mem.doNotOptimizeAway(state);
        printResult("mat_mul (Nx1)", iters, elapsedNs(io, start_ns));
    }

    std.debug.print("\n", .{});
}

fn printResult(name: []const u8, iters: usize, elapsed_ns: u64) void {
    const ns_per_op = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters));
    std.debug.print("  {s: <15}  {d: >8} iters   {d: >12.2} ns/op\n", .{ name, iters, ns_per_op });
}

fn nowNs(io: Io) i96 {
    return Io.Clock.awake.now(io).nanoseconds;
}

fn elapsedNs(io: Io, start_ns: i96) u64 {
    const end_ns = nowNs(io);
    return @intCast(end_ns - start_ns);
}

fn scaledIters(target_work: usize, scale: usize, min_iters: usize, max_iters: usize) usize {
    const denominator = if (scale == 0) 1 else scale;
    return clampUsize(target_work / denominator, min_iters, max_iters);
}

fn clampUsize(value: usize, min_value: usize, max_value: usize) usize {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

fn nextPerturb(state: *u64) T {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u16 = @truncate(state.* >> 48);
    return @as(T, @floatFromInt(bits + 1)) * 1e-8;
}

fn makeGeneralSquareData(comptime n: usize, comptime seed: usize) [n * n]T {
    var data: [n * n]T = undefined;
    for (0..n) |row| {
        for (0..n) |col| {
            if (row == col) {
                data[row * n + col] = @as(T, @floatFromInt(n * 12 + seed + 1));
            } else {
                const raw = ((row + seed * 3) * 17 + (col + seed * 5) * 13) % 7;
                data[row * n + col] = @as(T, @floatFromInt(raw + 1));
            }
        }
    }
    return data;
}

fn makeSpdSquareData(comptime n: usize) [n * n]T {
    var data = zeroData(n, n);

    for (0..n) |row| {
        var row_sum: T = 0;
        for (0..n) |col| {
            if (row == col) continue;

            const delta = if (row > col) row - col else col - row;
            const value = @as(T, 1) / @as(T, @floatFromInt(delta + 2));
            data[row * n + col] = value;
            row_sum += value;
        }
        data[row * n + row] = row_sum + @as(T, 1);
    }

    return data;
}

fn makeRectData(comptime rows: usize, comptime cols: usize, comptime seed: usize) [rows * cols]T {
    var data: [rows * cols]T = undefined;
    for (0..rows) |row| {
        for (0..cols) |col| {
            const raw = ((row + seed) * 19 + (col + seed * 2) * 11) % 13;
            data[row * cols + col] = @as(T, @floatFromInt(raw + 1));
        }
    }
    return data;
}

fn makeVectorData(comptime len: usize, comptime seed: usize) @Vector(len, T) {
    var data: [len]T = undefined;
    for (0..len) |idx| {
        const raw = ((idx + seed) * 7) % 9;
        data[idx] = @as(T, @floatFromInt(raw + 1));
    }
    return @as(@Vector(len, T), data);
}

fn zeroData(comptime rows: usize, comptime cols: usize) [rows * cols]T {
    return [_]T{@as(T, 0)} ** (rows * cols);
}
