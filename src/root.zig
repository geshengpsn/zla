//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

fn vectorChild(comptime Vec: type) type {
    return switch (@typeInfo(Vec)) {
        .vector => |info| info.child,
        else => @compileError("Expected a SIMD vector type"),
    };
}

fn vectorLen(comptime Vec: type) usize {
    return switch (@typeInfo(Vec)) {
        .vector => |info| info.len,
        else => @compileError("Expected a SIMD vector type"),
    };
}

fn assertFloat(comptime T: type, comptime message: []const u8) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(message),
    }
}

pub fn vec_dot(a: anytype, b: @TypeOf(a)) vectorChild(@TypeOf(a)) {
    comptime {
        _ = vectorLen(@TypeOf(a));
    }
    return @reduce(.Add, a * b);
}

pub fn vec_cross(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const Vec = @TypeOf(a);

    comptime {
        if (vectorLen(Vec) != 3) {
            @compileError("vec_cross expects @Vector(3, T) inputs");
        }
    }

    return @as(Vec, .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    });
}

pub fn Quaternion(comptime T: type) type {
    return struct {
        x: T,
        y: T,
        z: T,
        w: T,

        pub fn init(x: T, y: T, z: T, w: T) @This() {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub fn identity() @This() {
            return .{ .x = 0, .y = 0, .z = 0, .w = 1 };
        }

        pub fn norm(self: @This()) T {
            return std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w);
        }

        pub fn normalized(self: @This()) @This() {
            const n = self.norm();
            return .{
                .x = self.x / n,
                .y = self.y / n,
                .z = self.z / n,
                .w = self.w / n,
            };
        }

        pub fn toMat3(self: @This()) Mat(T, 3, 3) {
            comptime assertFloat(T, "Quaternion matrix conversion requires floating point element types");

            const q = self.normalized();
            const two: T = 2;

            const xx = q.x * q.x;
            const yy = q.y * q.y;
            const zz = q.z * q.z;
            const xy = q.x * q.y;
            const xz = q.x * q.z;
            const yz = q.y * q.z;
            const wx = q.w * q.x;
            const wy = q.w * q.y;
            const wz = q.w * q.z;

            return Mat(T, 3, 3).init(.{
                1 - two * (yy + zz), two * (xy - wz),     two * (xz + wy),
                two * (xy + wz),     1 - two * (xx + zz), two * (yz - wx),
                two * (xz - wy),     two * (yz + wx),     1 - two * (xx + yy),
            });
        }

        pub fn fromMat3(matrix: *const Mat(T, 3, 3)) @This() {
            comptime assertFloat(T, "Quaternion matrix conversion requires floating point element types");

            const m00 = matrix.data[0][0];
            const m01 = matrix.data[1][0];
            const m02 = matrix.data[2][0];
            const m10 = matrix.data[0][1];
            const m11 = matrix.data[1][1];
            const m12 = matrix.data[2][1];
            const m20 = matrix.data[0][2];
            const m21 = matrix.data[1][2];
            const m22 = matrix.data[2][2];

            const trace = m00 + m11 + m22;
            var q: @This() = undefined;

            if (trace > 0) {
                const s = std.math.sqrt(trace + 1) * 2;
                q = .{
                    .x = (m21 - m12) / s,
                    .y = (m02 - m20) / s,
                    .z = (m10 - m01) / s,
                    .w = s / 4,
                };
            } else if (m00 > m11 and m00 > m22) {
                const s = std.math.sqrt(1 + m00 - m11 - m22) * 2;
                q = .{
                    .x = s / 4,
                    .y = (m01 + m10) / s,
                    .z = (m02 + m20) / s,
                    .w = (m21 - m12) / s,
                };
            } else if (m11 > m22) {
                const s = std.math.sqrt(1 + m11 - m00 - m22) * 2;
                q = .{
                    .x = (m01 + m10) / s,
                    .y = s / 4,
                    .z = (m12 + m21) / s,
                    .w = (m02 - m20) / s,
                };
            } else {
                const s = std.math.sqrt(1 + m22 - m00 - m11) * 2;
                q = .{
                    .x = (m02 + m20) / s,
                    .y = (m12 + m21) / s,
                    .z = s / 4,
                    .w = (m10 - m01) / s,
                };
            }

            return q.normalized();
        }
    };
}

pub fn Mat(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [cols]@Vector(rows, T),
        comptime rows: usize = rows,
        comptime cols: usize = cols,

        pub fn init(data: [cols * rows]T) @This() {
            var self: @This() = .{
                .data = undefined,
            };
            inline for (0..cols) |j| {
                var col: [rows]T = undefined;
                inline for (0..rows) |i| {
                    col[i] = data[i * cols + j];
                }
                self.data[j] = @as(@Vector(rows, T), col);
            }
            return self;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            inline for (0..rows) |i| {
                inline for (0..cols) |j| {
                    try writer.print("{d:8.4} ", .{self.data[j][i]});
                }
                try writer.print("\n", .{});
            }
        }

        pub fn transpose(self: *const @This()) Mat(T, cols, rows) {
            var out: Mat(T, cols, rows) = .{
                .data = undefined,
            };
            inline for (0..rows) |out_col| {
                var col: [cols]T = undefined;
                inline for (0..cols) |out_row| {
                    col[out_row] = self.data[out_row][out_col];
                }
                out.data[out_col] = @as(@Vector(cols, T), col);
            }
            return out;
        }

        pub fn get(self: *const @This(), row: usize, col: usize) T {
            const col_data = @as([rows]T, self.data[col]);
            return col_data[row];
        }

        pub fn set(self: *@This(), row: usize, col: usize, value: T) void {
            var col_data = @as([rows]T, self.data[col]);
            col_data[row] = value;
            self.data[col] = @as(@Vector(rows, T), col_data);
        }

        pub fn get_row(self: *const @This(), row: usize) @Vector(cols, T) {
            var row_data: [cols]T = undefined;
            inline for (0..cols) |col| {
                const col_data = @as([rows]T, self.data[col]);
                row_data[col] = col_data[row];
            }
            return @as(@Vector(cols, T), row_data);
        }

        pub fn set_row(self: *@This(), row: usize, value: @Vector(cols, T)) void {
            const row_data = @as([cols]T, value);
            inline for (0..cols) |col| {
                var col_data = @as([rows]T, self.data[col]);
                col_data[row] = row_data[col];
                self.data[col] = @as(@Vector(rows, T), col_data);
            }
        }

        pub fn get_col(self: *const @This(), col: usize) @Vector(rows, T) {
            return self.data[col];
        }

        pub fn set_col(self: *@This(), col: usize, value: @Vector(rows, T)) void {
            self.data[col] = value;
        }

        pub fn get_block(
            self: *const @This(),
            comptime block_rows: usize,
            comptime block_cols: usize,
            start_row: usize,
            start_col: usize,
        ) Mat(T, block_rows, block_cols) {
            comptime {
                if (block_rows > rows or block_cols > cols) {
                    @compileError("Block dimensions exceed matrix dimensions");
                }
            }

            var out: Mat(T, block_rows, block_cols) = .{
                .data = undefined,
            };
            inline for (0..block_cols) |block_col| {
                var col_data: [block_rows]T = undefined;
                inline for (0..block_rows) |block_row| {
                    col_data[block_row] = self.get(start_row + block_row, start_col + block_col);
                }
                out.data[block_col] = @as(@Vector(block_rows, T), col_data);
            }
            return out;
        }

        pub fn set_block(self: *@This(), start_row: usize, start_col: usize, block: anytype) void {
            comptime {
                if (block.*.rows > rows or block.*.cols > cols) {
                    @compileError("Block dimensions exceed matrix dimensions");
                }
            }

            inline for (0..block.*.cols) |block_col| {
                inline for (0..block.*.rows) |block_row| {
                    self.set(
                        start_row + block_row,
                        start_col + block_col,
                        block.*.data[block_col][block_row],
                    );
                }
            }
        }

        pub fn mat_add(a: *const @This(), b: *const @This(), c: *@This()) void {
            inline for (0..cols) |col| {
                c.data[col] = a.data[col] + b.data[col];
            }
        }

        pub fn mat_add_assign(a: *@This(), b: *const @This()) void {
            inline for (0..cols) |col| {
                a.data[col] += b.data[col];
            }
        }

        pub fn mat_mul(a: *const @This(), b: anytype, c: anytype) void {
            comptime {
                if (b.*.rows != a.cols) {
                    @compileError("Matrix A and B dimensions do not match");
                }
                if (c.*.rows != a.rows or c.*.cols != b.*.cols) {
                    @compileError("Result matrix C dimensions do not match");
                }
            }
            inline for (0..b.*.cols) |i| {
                vec_mul(a, &b.*.data[i], &c.*.data[i]);
            }
        }

        pub fn mat_inv(a: *const @This(), out: *@This()) !void {
            comptime {
                if (rows != cols) {
                    @compileError("Matrix inversion requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("Matrix inversion requires floating point element types"),
                }
            }

            var left = a.to_row_major();
            var right = identity_row_major();
            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |pivot_col| {
                var pivot_row = pivot_col;
                var pivot_abs = abs_value(left[pivot_row][pivot_col]);

                inline for (pivot_col + 1..rows) |row| {
                    const candidate = abs_value(left[row][pivot_col]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = row;
                    }
                }

                if (pivot_abs <= eps) {
                    return error.SingularMatrix;
                }

                if (pivot_row != pivot_col) {
                    std.mem.swap(@Vector(cols, T), &left[pivot_col], &left[pivot_row]);
                    std.mem.swap(@Vector(cols, T), &right[pivot_col], &right[pivot_row]);
                }

                const pivot_value = left[pivot_col][pivot_col];
                const inv_pivot = @as(T, 1) / pivot_value;
                const inv_pivot_vec = @as(@Vector(cols, T), @splat(inv_pivot));
                left[pivot_col] *= inv_pivot_vec;
                right[pivot_col] *= inv_pivot_vec;

                inline for (0..rows) |row| {
                    if (row != pivot_col) {
                        const factor = left[row][pivot_col];
                        if (abs_value(factor) > eps) {
                            const factor_vec = @as(@Vector(cols, T), @splat(factor));
                            left[row] -= left[pivot_col] * factor_vec;
                            right[row] -= right[pivot_col] * factor_vec;
                        }
                    }
                }
            }

            out.* = from_row_major(right);
        }

        pub fn solve_lu(a: *const @This(), b: *const @Vector(rows, T), x: *@Vector(rows, T)) !void {
            comptime {
                if (rows != cols) {
                    @compileError("LU solve requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("LU solve requires floating point element types"),
                }
            }

            var lu = a.to_row_major();
            var permutation: [rows]usize = undefined;
            inline for (0..rows) |i| {
                permutation[i] = i;
            }

            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |pivot_col| {
                var pivot_row = pivot_col;
                var pivot_abs = abs_value(lu[pivot_row][pivot_col]);

                inline for (pivot_col + 1..rows) |row| {
                    const candidate = abs_value(lu[row][pivot_col]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = row;
                    }
                }

                if (pivot_abs <= eps) {
                    return error.SingularMatrix;
                }

                if (pivot_row != pivot_col) {
                    std.mem.swap(@Vector(cols, T), &lu[pivot_col], &lu[pivot_row]);
                    std.mem.swap(usize, &permutation[pivot_col], &permutation[pivot_row]);
                }

                const pivot = lu[pivot_col][pivot_col];
                const tail = range_mask(pivot_col + 1, cols);
                inline for (pivot_col + 1..rows) |row| {
                    const factor = lu[row][pivot_col] / pivot;
                    const factor_vec = @as(@Vector(cols, T), @splat(factor));
                    const updated = lu[row] - lu[pivot_col] * factor_vec;

                    lu[row] = @select(T, tail, updated, lu[row]);
                    lu[row][pivot_col] = factor;
                }
            }

            const rhs = @as([rows]T, b.*);
            var pb: [rows]T = undefined;
            inline for (0..rows) |row| {
                pb[row] = rhs[permutation[row]];
            }

            var y = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |row| {
                const known = dot_range(lu[row], y, 0, row);
                y[row] = pb[row] - known;
            }

            var result = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |idx| {
                const row = rows - 1 - idx;
                const diag = lu[row][row];
                if (abs_value(diag) <= eps) {
                    return error.SingularMatrix;
                }

                const known = dot_range(lu[row], result, row + 1, cols);
                result[row] = (y[row] - known) / diag;
            }

            x.* = result;
        }

        pub fn solve_ldlt(a: *const @This(), b: *const @Vector(rows, T), x: *@Vector(rows, T)) !void {
            comptime {
                if (rows != cols) {
                    @compileError("LDLT solve requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("LDLT solve requires floating point element types"),
                }
            }

            var matrix: [rows][cols]T = undefined;
            inline for (0..rows) |row| {
                inline for (0..cols) |col| {
                    matrix[row][col] = a.data[col][row];
                }
            }
            var l = std.mem.zeroes([rows][cols]T);
            var d = [_]T{0} ** rows;

            const eps = std.math.floatEps(T) * @as(T, 16);

            for (0..rows) |col| {
                var diag = matrix[col][col];
                for (0..col) |idx| {
                    diag -= l[col][idx] * l[col][idx] * d[idx];
                }

                if (abs_value(diag) <= eps) {
                    return error.SingularMatrix;
                }

                d[col] = diag;
                l[col][col] = @as(T, 1);

                for (col + 1..rows) |row| {
                    var value = matrix[row][col];
                    for (0..col) |idx| {
                        value -= l[row][idx] * d[idx] * l[col][idx];
                    }
                    l[row][col] = value / diag;
                }
            }

            const rhs = @as([rows]T, b.*);
            var y = [_]T{0} ** rows;
            for (0..rows) |row| {
                var value = rhs[row];
                for (0..row) |col| {
                    value -= l[row][col] * y[col];
                }
                y[row] = value;
            }

            var z = [_]T{0} ** rows;
            for (0..rows) |row| {
                z[row] = y[row] / d[row];
            }

            var result = [_]T{0} ** rows;
            for (0..rows) |idx| {
                const row = rows - 1 - idx;
                var value = z[row];
                for (row + 1..rows) |col| {
                    value -= l[col][row] * result[col];
                }
                result[row] = value;
            }

            x.* = @as(@Vector(rows, T), result);
        }

        pub fn solve_cholesky(a: *const @This(), b: *const @Vector(rows, T), x: *@Vector(rows, T)) !void {
            comptime {
                if (rows != cols) {
                    @compileError("Cholesky solve requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("Cholesky solve requires floating point element types"),
                }
            }

            const matrix = a.to_row_major();
            var l = std.mem.zeroes([rows]@Vector(cols, T));

            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |row| {
                inline for (0..row + 1) |col| {
                    var sum = matrix[row][col];
                    if (col > 0) {
                        sum -= dot_range(l[row], l[col], 0, col);
                    }

                    if (row == col) {
                        if (sum <= eps) {
                            return error.NotPositiveDefinite;
                        }
                        l[row][col] = std.math.sqrt(sum);
                    } else {
                        const diag = l[col][col];
                        if (abs_value(diag) <= eps) {
                            return error.NotPositiveDefinite;
                        }
                        l[row][col] = sum / diag;
                    }
                }
            }

            var y = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |row| {
                const diag = l[row][row];
                if (abs_value(diag) <= eps) {
                    return error.NotPositiveDefinite;
                }

                const known = dot_range(l[row], y, 0, row);
                y[row] = (b[row] - known) / diag;
            }

            const lt = transpose_square_rows(l);
            var result = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |idx| {
                const row = rows - 1 - idx;
                const diag = lt[row][row];
                if (abs_value(diag) <= eps) {
                    return error.NotPositiveDefinite;
                }

                const known = dot_range(lt[row], result, row + 1, cols);
                result[row] = (y[row] - known) / diag;
            }

            x.* = result;
        }

        pub fn vec_mul(a: *const @This(), b: *const @Vector(a.cols, T), c: *@Vector(a.rows, T)) void {
            c.* = @splat(0);
            inline for (0..a.cols) |i| {
                c.* += a.data[i] * @as(@Vector(a.rows, T), @splat(b[i]));
            }
        }

        fn to_row_major(self: *const @This()) [rows]@Vector(cols, T) {
            var matrix: [rows]@Vector(cols, T) = undefined;
            inline for (0..rows) |row| {
                var row_data: [cols]T = undefined;
                inline for (0..cols) |col| {
                    row_data[col] = self.data[col][row];
                }
                matrix[row] = @as(@Vector(cols, T), row_data);
            }
            return matrix;
        }

        fn from_row_major(matrix: [rows]@Vector(cols, T)) @This() {
            var self: @This() = .{
                .data = undefined,
            };
            inline for (0..cols) |col| {
                var col_data: [rows]T = undefined;
                inline for (0..rows) |row| {
                    col_data[row] = matrix[row][col];
                }
                self.data[col] = @as(@Vector(rows, T), col_data);
            }
            return self;
        }

        fn identity_row_major() [rows]@Vector(cols, T) {
            comptime {
                if (rows != cols) {
                    @compileError("identity_row_major requires square matrices");
                }
            }

            var matrix = std.mem.zeroes([rows]@Vector(cols, T));
            inline for (0..rows) |idx| {
                matrix[idx][idx] = @as(T, 1);
            }
            return matrix;
        }

        fn transpose_square_rows(matrix: [rows]@Vector(cols, T)) [rows]@Vector(cols, T) {
            comptime {
                if (rows != cols) {
                    @compileError("transpose_square_rows requires square matrices");
                }
            }

            var transposed: [rows]@Vector(cols, T) = undefined;
            inline for (0..rows) |row| {
                var row_data: [cols]T = undefined;
                inline for (0..cols) |col| {
                    row_data[col] = matrix[col][row];
                }
                transposed[row] = @as(@Vector(cols, T), row_data);
            }
            return transposed;
        }

        fn dot_range(
            lhs: @Vector(cols, T),
            rhs: @Vector(cols, T),
            comptime start: usize,
            comptime end: usize,
        ) T {
            comptime {
                if (start > end or end > cols) {
                    @compileError("Invalid dot product range");
                }
            }

            if (start == end) {
                return @as(T, 0);
            }

            const mask = range_mask(start, end);
            const product = lhs * rhs;
            const zero = @as(@Vector(cols, T), @splat(@as(T, 0)));
            return @reduce(.Add, @select(T, mask, product, zero));
        }

        fn range_mask(comptime start: usize, comptime end: usize) @Vector(cols, bool) {
            comptime {
                if (start > end or end > cols) {
                    @compileError("Invalid mask range");
                }
            }

            var mask = [_]bool{false} ** cols;
            inline for (start..end) |idx| {
                mask[idx] = true;
            }
            return @as(@Vector(cols, bool), mask);
        }

        fn abs_value(value: T) T {
            return switch (@typeInfo(T)) {
                .float => @abs(value),
                else => unreachable,
            };
        }
    };
}

test "vec_dot" {
    const a = @Vector(4, f64){ 1, 2, 3, 4 };
    const b = @Vector(4, f64){ 5, 6, 7, 8 };
    try std.testing.expectApproxEqAbs(70, vec_dot(a, b), 1e-12);
}

test "vec_cross" {
    const x = @Vector(3, f64){ 1, 0, 0 };
    const y = @Vector(3, f64){ 0, 1, 0 };
    const z = vec_cross(x, y);
    try std.testing.expectEqual(@Vector(3, f64){ 0, 0, 1 }, z);
}

test "vec_cross orthogonality" {
    const a = @Vector(3, f64){ 2, -1, 3 };
    const b = @Vector(3, f64){ -4, 5, 1 };
    const c = vec_cross(a, b);

    try std.testing.expectApproxEqAbs(0, vec_dot(c, a), 1e-12);
    try std.testing.expectApproxEqAbs(0, vec_dot(c, b), 1e-12);
}

test "Quaternion toMat3 z rotation" {
    const half_sqrt = std.math.sqrt(@as(f64, 0.5));
    const q = Quaternion(f64).init(0, 0, half_sqrt, half_sqrt);
    const m = q.toMat3();

    try std.testing.expectApproxEqAbs(0, m.data[0][0], 1e-12);
    try std.testing.expectApproxEqAbs(1, m.data[0][1], 1e-12);
    try std.testing.expectApproxEqAbs(0, m.data[0][2], 1e-12);

    try std.testing.expectApproxEqAbs(-1, m.data[1][0], 1e-12);
    try std.testing.expectApproxEqAbs(0, m.data[1][1], 1e-12);
    try std.testing.expectApproxEqAbs(0, m.data[1][2], 1e-12);

    try std.testing.expectApproxEqAbs(0, m.data[2][0], 1e-12);
    try std.testing.expectApproxEqAbs(0, m.data[2][1], 1e-12);
    try std.testing.expectApproxEqAbs(1, m.data[2][2], 1e-12);
}

test "Quaternion fromMat3 round trip" {
    const half_sqrt = std.math.sqrt(@as(f64, 0.5));
    const original = Quaternion(f64).init(0.5, -0.5, 0.5, half_sqrt).normalized();
    const matrix = original.toMat3();
    const round_tripped = Quaternion(f64).fromMat3(&matrix);
    const round_tripped_matrix = round_tripped.toMat3();

    inline for (0..3) |col| {
        inline for (0..3) |row| {
            try std.testing.expectApproxEqAbs(matrix.data[col][row], round_tripped_matrix.data[col][row], 1e-12);
        }
    }
}

test "Mat init" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    try std.testing.expectEqual(a.data[0], @Vector(2, f32){ 1, 4 });
    try std.testing.expectEqual(a.data[1], @Vector(2, f32){ 2, 5 });
    try std.testing.expectEqual(a.data[2], @Vector(2, f32){ 3, 6 });
}

test "Mat format" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    var buf = try std.ArrayList(u8).initCapacity(std.testing.allocator, 100);
    defer buf.deinit(std.testing.allocator);
    var writer = std.Io.Writer.fromArrayList(&buf);
    try writer.print("{f}", .{a});
    buf = writer.toArrayList();
    try std.testing.expectEqualStrings("  1.0000   2.0000   3.0000 \n  4.0000   5.0000   6.0000 \n", buf.items);
}

test "Mat mat_mul" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    const b = Mat(f32, 3, 2).init(.{
        1, 2,
        3, 4,
        5, 6,
    });

    var c = Mat(f32, 2, 2).init(.{ 0, 0, 0, 0 });
    a.mat_mul(&b, &c);

    const expected = Mat(f32, 2, 2).init(.{
        22, 28,
        49, 64,
    });
    try std.testing.expectEqual(c, expected);
}

test "Mat mat_add" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    const b = Mat(f32, 2, 3).init(.{
        6, 5, 4,
        3, 2, 1,
    });

    var c = Mat(f32, 2, 3).init(.{ 0, 0, 0, 0, 0, 0 });
    a.mat_add(&b, &c);

    const expected = Mat(f32, 2, 3).init(.{
        7, 7, 7,
        7, 7, 7,
    });
    try std.testing.expectEqual(expected, c);
}

test "Mat mat_add_assign" {
    var a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    const b = Mat(f32, 2, 3).init(.{
        6, 5, 4,
        3, 2, 1,
    });

    a.mat_add_assign(&b);

    const expected = Mat(f32, 2, 3).init(.{
        7, 7, 7,
        7, 7, 7,
    });
    try std.testing.expectEqual(expected, a);
}

test "Mat transpose" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    const transposed = a.transpose();

    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, transposed.data[0]);
    try std.testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, transposed.data[1]);
}

test "Mat get and set element" {
    var a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    try std.testing.expectEqual(@as(f32, 5), a.get(1, 1));

    a.set(0, 2, 9);
    try std.testing.expectEqual(@as(f32, 9), a.get(0, 2));
    try std.testing.expectEqual(@Vector(2, f32){ 9, 6 }, a.data[2]);
}

test "Mat get and set row" {
    var a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    try std.testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, a.get_row(1));

    a.set_row(0, @Vector(3, f32){ 7, 8, 9 });
    try std.testing.expectEqual(@Vector(3, f32){ 7, 8, 9 }, a.get_row(0));
    try std.testing.expectEqual(@Vector(2, f32){ 7, 4 }, a.data[0]);
}

test "Mat get and set col" {
    var a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    try std.testing.expectEqual(@Vector(2, f32){ 2, 5 }, a.get_col(1));

    a.set_col(1, @Vector(2, f32){ 8, 9 });
    try std.testing.expectEqual(@Vector(2, f32){ 8, 9 }, a.get_col(1));
    try std.testing.expectEqual(@as(f32, 8), a.get(0, 1));
}

test "Mat get and set block" {
    var a = Mat(f32, 3, 4).init(.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });

    const block = a.get_block(2, 2, 1, 1);
    const expected = Mat(f32, 2, 2).init(.{
        6,  7,
        10, 11,
    });
    try std.testing.expectEqual(expected, block);

    const replacement = Mat(f32, 2, 2).init(.{
        20, 21,
        22, 23,
    });
    a.set_block(0, 2, &replacement);

    try std.testing.expectEqual(@as(f32, 20), a.get(0, 2));
    try std.testing.expectEqual(@as(f32, 21), a.get(0, 3));
    try std.testing.expectEqual(@as(f32, 22), a.get(1, 2));
    try std.testing.expectEqual(@as(f32, 23), a.get(1, 3));
}

test "Mat mat_inv" {
    const a = Mat(f64, 2, 2).init(.{
        4, 7,
        2, 6,
    });

    var inv = Mat(f64, 2, 2).init(.{ 0, 0, 0, 0 });
    try a.mat_inv(&inv);

    try std.testing.expectApproxEqAbs(0.6, inv.data[0][0], 1e-10);
    try std.testing.expectApproxEqAbs(-0.2, inv.data[0][1], 1e-10);
    try std.testing.expectApproxEqAbs(-0.7, inv.data[1][0], 1e-10);
    try std.testing.expectApproxEqAbs(0.4, inv.data[1][1], 1e-10);
}

test "Mat mat_inv multiply identity" {
    const a = Mat(f64, 3, 3).init(.{
        1, 2, 3,
        0, 1, 4,
        5, 6, 0,
    });

    var inv = Mat(f64, 3, 3).init(.{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try a.mat_inv(&inv);

    var product = Mat(f64, 3, 3).init(.{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    a.mat_mul(&inv, &product);

    const identity = Mat(f64, 3, 3).init(.{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    });

    inline for (0..3) |col| {
        inline for (0..3) |row| {
            try std.testing.expectApproxEqAbs(identity.data[col][row], product.data[col][row], 1e-10);
        }
    }
}

test "Mat mat_inv singular" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, 4,
    });

    var inv = Mat(f64, 2, 2).init(.{ 0, 0, 0, 0 });
    try std.testing.expectError(error.SingularMatrix, a.mat_inv(&inv));
}

test "Mat solve_lu" {
    const a = Mat(f64, 3, 3).init(.{
        3,  2,   -1,
        2,  -2,  4,
        -1, 0.5, -1,
    });
    const b = @Vector(3, f64){ 1, -2, 0 };

    var x = @as(@Vector(3, f64), @splat(0));
    try a.solve_lu(&b, &x);

    const expected = @Vector(3, f64){ 1, -2, -2 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
    }
}

test "Mat solve_lu singular" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, 4,
    });
    const b = @Vector(2, f64){ 1, 2 };

    var x = @as(@Vector(2, f64), @splat(0));
    try std.testing.expectError(error.SingularMatrix, a.solve_lu(&b, &x));
}

test "Mat solve_ldlt" {
    const a = Mat(f64, 3, 3).init(.{
        4, 1, 1,
        1, 3, 0,
        1, 0, 2,
    });
    const b = @Vector(3, f64){ 9, 7, 7 };

    var x = @as(@Vector(3, f64), @splat(0));
    try a.solve_ldlt(&b, &x);

    const expected = @Vector(3, f64){ 1, 2, 3 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
    }
}

test "Mat solve_ldlt indefinite" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, -3,
    });
    const b = @Vector(2, f64){ 5, -4 };

    var x = @as(@Vector(2, f64), @splat(0));
    try a.solve_ldlt(&b, &x);

    const expected = @Vector(2, f64){ 1, 2 };
    inline for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
    }
}

test "Mat solve_ldlt singular" {
    const a = Mat(f64, 2, 2).init(.{
        0, 0,
        0, 1,
    });
    const b = @Vector(2, f64){ 1, 1 };

    var x = @as(@Vector(2, f64), @splat(0));
    try std.testing.expectError(error.SingularMatrix, a.solve_ldlt(&b, &x));
}

test "Mat solve_cholesky" {
    const a = Mat(f64, 3, 3).init(.{
        4, 1, 1,
        1, 3, 0,
        1, 0, 2,
    });
    const b = @Vector(3, f64){ 9, 7, 7 };

    var x = @as(@Vector(3, f64), @splat(0));
    try a.solve_cholesky(&b, &x);

    const expected = @Vector(3, f64){ 1, 2, 3 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
    }
}

test "Mat solve_cholesky not positive definite" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, 1,
    });
    const b = @Vector(2, f64){ 1, 1 };

    var x = @as(@Vector(2, f64), @splat(0));
    try std.testing.expectError(error.NotPositiveDefinite, a.solve_cholesky(&b, &x));
}
