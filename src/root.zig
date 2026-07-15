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

pub fn vecDot(a: anytype, b: @TypeOf(a)) vectorChild(@TypeOf(a)) {
    var out: vectorChild(@TypeOf(a)) = undefined;
    vecDotAssign(a, b, &out);
    return out;
}

pub fn vecDotAssign(a: anytype, b: @TypeOf(a), out: *vectorChild(@TypeOf(a))) void {
    comptime {
        _ = vectorLen(@TypeOf(a));
    }
    out.* = @reduce(.Add, a * b);
}

pub fn vecCross(a: @Vector(3, f64), b: @Vector(3, f64)) @Vector(3, f64) {
    var out: @Vector(3, f64) = undefined;
    vecCrossAssign(a, b, &out);
    return out;
}

pub fn vecCrossAssign(a: @Vector(3, f64), b: @Vector(3, f64), out: *@Vector(3, f64)) void {
    out.* = .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn vecScale(a: anytype, b: vectorChild(@TypeOf(a))) @TypeOf(a) {
    var out: @TypeOf(a) = undefined;
    vecScaleAssign(a, b, &out);
    return out;
}

pub fn vecScaleAssign(a: anytype, b: vectorChild(@TypeOf(a)), out: *@TypeOf(a)) void {
    out.* = @as(@TypeOf(a), @splat(b)) * a;
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
            var out: T = undefined;
            self.normAssign(&out);
            return out;
        }

        pub fn normAssign(self: *const @This(), out: *T) void {
            out.* = std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w);
        }

        pub fn normalized(self: @This()) @This() {
            var out: @This() = undefined;
            self.normalizedAssign(&out);
            return out;
        }

        pub fn normalizedAssign(self: *const @This(), out: *@This()) void {
            const n = self.norm();
            out.x = self.x / n;
            out.y = self.y / n;
            out.z = self.z / n;
            out.w = self.w / n;
        }

        pub fn add(a: @This(), b: @This()) @This() {
            var out: @This() = undefined;
            a.addAssign(&b, &out);
            return out;
        }

        pub fn addAssign(a: *const @This(), b: *const @This(), out: *@This()) void {
            out.x = a.x + b.x;
            out.y = a.y + b.y;
            out.z = a.z + b.z;
            out.w = a.w + b.w;
        }

        pub fn conjugate(self: @This()) @This() {
            var out: @This() = undefined;
            self.conjugateAssign(&out);
            return out;
        }

        pub fn conjugateAssign(self: *const @This(), out: *@This()) void {
            out.x = -self.x;
            out.y = -self.y;
            out.z = -self.z;
            out.w = self.w;
        }

        pub fn mul(a: @This(), b: @This()) @This() {
            var out: @This() = undefined;
            a.mulAssign(&b, &out);
            return out;
        }

        pub fn mulAssign(self: *const @This(), rhs: *const @This(), out: *@This()) void {
            const x = self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y;
            const y = self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x;
            const z = self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w;
            const w = self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z;
            out.* = .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub fn rotateVector(self: @This(), v: @Vector(3, T)) @Vector(3, T) {
            var out: @Vector(3, T) = undefined;
            self.rotateVectorAssign(&v, &out);
            return out;
        }

        pub fn rotateVectorAssign(self: *const @This(), v: *const @Vector(3, T), out: *@Vector(3, T)) void {
            comptime assertFloat(T, "Quaternion vector rotation requires floating point element types");

            const q = self.normalized();
            const qv = @Vector(3, T){ q.x, q.y, q.z };
            const t = vecCross(qv, v.*) * @as(@Vector(3, T), @splat(@as(T, 2)));
            out.* = v.* + t * @as(@Vector(3, T), @splat(q.w)) + vecCross(qv, t);
        }

        pub fn dot(self: @This(), omega: @Vector(3, T)) @This() {
            var out: @This() = undefined;
            self.dotAssign(&omega, &out);
            return out;
        }

        pub fn dotAssign(self: *const @This(), omega: *const @Vector(3, T), out: *@This()) void {
            comptime assertFloat(T, "Quaternion angular velocity derivative requires floating point element types");

            const half: T = 0.5;
            const x = (self.w * omega[0] + self.y * omega[2] - self.z * omega[1]) * half;
            const y = (self.w * omega[1] - self.x * omega[2] + self.z * omega[0]) * half;
            const z = (self.w * omega[2] + self.x * omega[1] - self.y * omega[0]) * half;
            const w = -(self.x * omega[0] + self.y * omega[1] + self.z * omega[2]) * half;
            out.* = .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub fn toMat3(self: @This()) Mat(T, 3, 3) {
            var out: Mat(T, 3, 3) = .{ .data = undefined };
            self.toMat3Assign(&out);
            return out;
        }

        pub fn toMat3Assign(self: *const @This(), out: *Mat(T, 3, 3)) void {
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

            out.data[0] = .{
                1 - two * (yy + zz),
                two * (xy + wz),
                two * (xz - wy),
            };
            out.data[1] = .{
                two * (xy - wz),
                1 - two * (xx + zz),
                two * (yz + wx),
            };
            out.data[2] = .{
                two * (xz + wy),
                two * (yz - wx),
                1 - two * (xx + yy),
            };
        }

        pub fn fromMat3(matrix: *const Mat(T, 3, 3)) @This() {
            var out: @This() = undefined;
            fromMat3Assign(matrix, &out);
            return out;
        }

        pub fn fromMat3Assign(matrix: *const Mat(T, 3, 3), out: *@This()) void {
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
            if (trace > 0) {
                const s = std.math.sqrt(trace + 1) * 2;
                out.* = .{
                    .x = (m21 - m12) / s,
                    .y = (m02 - m20) / s,
                    .z = (m10 - m01) / s,
                    .w = s / 4,
                };
            } else if (m00 > m11 and m00 > m22) {
                const s = std.math.sqrt(1 + m00 - m11 - m22) * 2;
                out.* = .{
                    .x = s / 4,
                    .y = (m01 + m10) / s,
                    .z = (m02 + m20) / s,
                    .w = (m21 - m12) / s,
                };
            } else if (m11 > m22) {
                const s = std.math.sqrt(1 + m11 - m00 - m22) * 2;
                out.* = .{
                    .x = (m01 + m10) / s,
                    .y = s / 4,
                    .z = (m12 + m21) / s,
                    .w = (m02 - m20) / s,
                };
            } else {
                const s = std.math.sqrt(1 + m22 - m00 - m11) * 2;
                out.* = .{
                    .x = (m02 + m20) / s,
                    .y = (m12 + m21) / s,
                    .z = s / 4,
                    .w = (m10 - m01) / s,
                };
            }

            out.normalizedAssign(out);
        }
    };
}

pub fn Mat(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [cols]@Vector(rows, T),
        comptime rows: usize = rows,
        comptime cols: usize = cols,

        pub fn init(data: [cols * rows]T) @This() {
            var out: @This() = .{ .data = undefined };
            initAssign(&data, &out);
            return out;
        }

        pub fn initAssign(data: *const [cols * rows]T, out: *@This()) void {
            inline for (0..cols) |j| {
                var col: [rows]T = undefined;
                inline for (0..rows) |i| {
                    col[i] = data[i * cols + j];
                }
                out.data[j] = @as(@Vector(rows, T), col);
            }
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
            var out: Mat(T, cols, rows) = .{ .data = undefined };
            self.transposeAssign(&out);
            return out;
        }

        pub fn transposeAssign(self: *const @This(), out: *Mat(T, cols, rows)) void {
            inline for (0..rows) |out_col| {
                var col: [cols]T = undefined;
                inline for (0..cols) |out_row| {
                    col[out_row] = self.data[out_row][out_col];
                }
                out.data[out_col] = @as(@Vector(cols, T), col);
            }
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

        pub fn getRow(self: *const @This(), row: usize) @Vector(cols, T) {
            var row_data: [cols]T = undefined;
            inline for (0..cols) |col| {
                const col_data = @as([rows]T, self.data[col]);
                row_data[col] = col_data[row];
            }
            return @as(@Vector(cols, T), row_data);
        }

        pub fn setRow(self: *@This(), row: usize, value: @Vector(cols, T)) void {
            const row_data = @as([cols]T, value);
            inline for (0..cols) |col| {
                var col_data = @as([rows]T, self.data[col]);
                col_data[row] = row_data[col];
                self.data[col] = @as(@Vector(rows, T), col_data);
            }
        }

        pub fn getCol(self: *const @This(), col: usize) @Vector(rows, T) {
            return self.data[col];
        }

        pub fn setCol(self: *@This(), col: usize, value: @Vector(rows, T)) void {
            self.data[col] = value;
        }

        pub fn getBlock(
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

        pub fn setBlock(self: *@This(), start_row: usize, start_col: usize, block: anytype) void {
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

        pub fn matAdd(a: *const @This(), b: *const @This()) @This() {
            var out: @This() = .{ .data = undefined };
            a.matAddAssign(b, &out);
            return out;
        }

        pub fn matAddAssign(a: *const @This(), b: *const @This(), out: *@This()) void {
            inline for (0..cols) |col| {
                out.data[col] = a.data[col] + b.data[col];
            }
        }

        pub fn matNeg(a: *const @This()) @This() {
            var out: @This() = .{ .data = undefined };
            a.matNegAssign(&out);
            return out;
        }

        pub fn matNegAssign(a: *const @This(), out: *@This()) void {
            inline for (0..cols) |col| {
                out.data[col] = -a.data[col];
            }
        }

        pub fn matMul(a: *const @This(), b: anytype) Mat(T, rows, b.*.cols) {
            var out: Mat(T, rows, b.*.cols) = .{ .data = undefined };
            a.matMulAssign(b, &out);
            return out;
        }

        pub fn matMulAssign(a: *const @This(), b: anytype, out: anytype) void {
            comptime {
                if (b.*.rows != a.cols) {
                    @compileError("Matrix A and B dimensions do not match");
                }
                if (out.*.rows != a.rows or out.*.cols != b.*.cols) {
                    @compileError("Result matrix C dimensions do not match");
                }
            }
            inline for (0..b.*.cols) |i| {
                vecMulAssign(a, &b.*.data[i], &out.*.data[i]);
            }
        }

        pub fn inverse(self: *const @This()) !@This() {
            var out: @This() = .{ .data = undefined };
            try self.inverseAssign(&out);
            return out;
        }

        pub fn inverseAssign(a: *const @This(), out: *@This()) !void {
            comptime {
                if (rows != cols) {
                    @compileError("Matrix inversion requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("Matrix inversion requires floating point element types"),
                }
            }

            var left = a.toRowMajor();
            var right = identityRowMajor();
            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |pivot_col| {
                var pivot_row = pivot_col;
                var pivot_abs = absValue(left[pivot_row][pivot_col]);

                inline for (pivot_col + 1..rows) |row| {
                    const candidate = absValue(left[row][pivot_col]);
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
                        if (absValue(factor) > eps) {
                            const factor_vec = @as(@Vector(cols, T), @splat(factor));
                            left[row] -= left[pivot_col] * factor_vec;
                            right[row] -= right[pivot_col] * factor_vec;
                        }
                    }
                }
            }

            out.* = fromRowMajor(right);
        }

        pub fn solveLu(a: *const @This(), b: *const @Vector(rows, T)) !@Vector(rows, T) {
            var out: @Vector(rows, T) = undefined;
            try a.solveLuAssign(b, &out);
            return out;
        }

        pub fn solveLuAssign(a: *const @This(), b: *const @Vector(rows, T), out: *@Vector(rows, T)) !void {
            comptime {
                if (rows != cols) {
                    @compileError("LU solve requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("LU solve requires floating point element types"),
                }
            }

            var lu = a.toRowMajor();
            var permutation: [rows]usize = undefined;
            inline for (0..rows) |i| {
                permutation[i] = i;
            }

            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |pivot_col| {
                var pivot_row = pivot_col;
                var pivot_abs = absValue(lu[pivot_row][pivot_col]);

                inline for (pivot_col + 1..rows) |row| {
                    const candidate = absValue(lu[row][pivot_col]);
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
                const tail = rangeMask(pivot_col + 1, cols);
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
                const known = dotRange(lu[row], y, 0, row);
                y[row] = pb[row] - known;
            }

            var result = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |idx| {
                const row = rows - 1 - idx;
                const diag = lu[row][row];
                if (absValue(diag) <= eps) {
                    return error.SingularMatrix;
                }

                const known = dotRange(lu[row], result, row + 1, cols);
                result[row] = (y[row] - known) / diag;
            }

            out.* = result;
        }

        pub fn solveLdlt(a: *const @This(), b: *const @Vector(rows, T)) !@Vector(rows, T) {
            var out: @Vector(rows, T) = undefined;
            try a.solveLdltAssign(b, &out);
            return out;
        }

        pub fn solveLdltAssign(a: *const @This(), b: *const @Vector(rows, T), out: *@Vector(rows, T)) !void {
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

                if (absValue(diag) <= eps) {
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

            out.* = @as(@Vector(rows, T), result);
        }

        pub fn solveCholesky(a: *const @This(), b: *const @Vector(rows, T)) !@Vector(rows, T) {
            var out: @Vector(rows, T) = undefined;
            try a.solveCholeskyAssign(b, &out);
            return out;
        }

        pub fn solveCholeskyAssign(a: *const @This(), b: *const @Vector(rows, T), out: *@Vector(rows, T)) !void {
            comptime {
                if (rows != cols) {
                    @compileError("Cholesky solve requires square matrices");
                }
                switch (@typeInfo(T)) {
                    .float => {},
                    else => @compileError("Cholesky solve requires floating point element types"),
                }
            }

            const matrix = a.toRowMajor();
            var l = std.mem.zeroes([rows]@Vector(cols, T));

            const eps = std.math.floatEps(T) * @as(T, 16);

            inline for (0..rows) |row| {
                inline for (0..row + 1) |col| {
                    var sum = matrix[row][col];
                    if (col > 0) {
                        sum -= dotRange(l[row], l[col], 0, col);
                    }

                    if (row == col) {
                        if (sum <= eps) {
                            return error.NotPositiveDefinite;
                        }
                        l[row][col] = std.math.sqrt(sum);
                    } else {
                        const diag = l[col][col];
                        if (absValue(diag) <= eps) {
                            return error.NotPositiveDefinite;
                        }
                        l[row][col] = sum / diag;
                    }
                }
            }

            var y = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |row| {
                const diag = l[row][row];
                if (absValue(diag) <= eps) {
                    return error.NotPositiveDefinite;
                }

                const known = dotRange(l[row], y, 0, row);
                y[row] = (b[row] - known) / diag;
            }

            const lt = transposeSquareRows(l);
            var result = @as(@Vector(rows, T), @splat(0));
            inline for (0..rows) |idx| {
                const row = rows - 1 - idx;
                const diag = lt[row][row];
                if (absValue(diag) <= eps) {
                    return error.NotPositiveDefinite;
                }

                const known = dotRange(lt[row], result, row + 1, cols);
                result[row] = (y[row] - known) / diag;
            }

            out.* = result;
        }

        pub fn vecMul(a: *const @This(), b: *const @Vector(a.cols, T)) @Vector(a.rows, T) {
            var out: @Vector(a.rows, T) = undefined;
            a.vecMulAssign(b, &out);
            return out;
        }

        pub fn vecMulAssign(a: *const @This(), b: *const @Vector(a.cols, T), out: *@Vector(a.rows, T)) void {
            out.* = @splat(0);
            inline for (0..a.cols) |i| {
                out.* += a.data[i] * @as(@Vector(a.rows, T), @splat(b[i]));
            }
        }

        fn toRowMajor(self: *const @This()) [rows]@Vector(cols, T) {
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

        fn fromRowMajor(matrix: [rows]@Vector(cols, T)) @This() {
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

        fn identityRowMajor() [rows]@Vector(cols, T) {
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

        fn transposeSquareRows(matrix: [rows]@Vector(cols, T)) [rows]@Vector(cols, T) {
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

        fn dotRange(
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

            const mask = rangeMask(start, end);
            const product = lhs * rhs;
            const zero = @as(@Vector(cols, T), @splat(@as(T, 0)));
            return @reduce(.Add, @select(T, mask, product, zero));
        }

        fn rangeMask(comptime start: usize, comptime end: usize) @Vector(cols, bool) {
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

        fn absValue(value: T) T {
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
    try std.testing.expectApproxEqAbs(70, vecDot(a, b), 1e-12);

    var assigned: f64 = undefined;
    vecDotAssign(a, b, &assigned);
    try std.testing.expectApproxEqAbs(70, assigned, 1e-12);
}

test "vec_cross" {
    const x = @Vector(3, f64){ 1, 0, 0 };
    const y = @Vector(3, f64){ 0, 1, 0 };
    const z = vecCross(x, y);
    try std.testing.expectEqual(@Vector(3, f64){ 0, 0, 1 }, z);

    var assigned: @Vector(3, f64) = undefined;
    vecCrossAssign(x, y, &assigned);
    try std.testing.expectEqual(z, assigned);
}

test "vec_cross orthogonality" {
    const a = @Vector(3, f64){ 2, -1, 3 };
    const b = @Vector(3, f64){ -4, 5, 1 };
    const c = vecCross(a, b);

    try std.testing.expectApproxEqAbs(0, vecDot(c, a), 1e-12);
    try std.testing.expectApproxEqAbs(0, vecDot(c, b), 1e-12);
}

test "vecScale" {
    const v = @Vector(4, f64){ 1, -2, 3, -4 };
    const scaled = vecScale(v, 0.5);

    try std.testing.expectEqual(@Vector(4, f64){ 0.5, -1, 1.5, -2 }, scaled);

    var assigned: @Vector(4, f64) = undefined;
    vecScaleAssign(v, 0.5, &assigned);
    try std.testing.expectEqual(scaled, assigned);
}

test "vecScale negative scalar" {
    const v = @Vector(3, i32){ 2, -3, 4 };
    const scaled = vecScale(v, -2);

    try std.testing.expectEqual(@Vector(3, i32){ -4, 6, -8 }, scaled);
}

test "Quaternion norm Assign variants" {
    const q = Quaternion(f64).init(1, 2, 2, 4);

    var norm: f64 = undefined;
    q.normAssign(&norm);
    try std.testing.expectApproxEqAbs(q.norm(), norm, 1e-12);

    var normalized: Quaternion(f64) = undefined;
    q.normalizedAssign(&normalized);
    const expected = q.normalized();
    try std.testing.expectEqual(expected, normalized);
}

test "Quaternion add" {
    const a = Quaternion(f64).init(1, 2, 3, 4);
    const b = Quaternion(f64).init(5, 6, 7, 8);
    const sum = a.add(b);

    try std.testing.expectApproxEqAbs(6, sum.x, 1e-12);
    try std.testing.expectApproxEqAbs(8, sum.y, 1e-12);
    try std.testing.expectApproxEqAbs(10, sum.z, 1e-12);
    try std.testing.expectApproxEqAbs(12, sum.w, 1e-12);
}

test "Quaternion addAssign" {
    const a = Quaternion(f64).init(1, 2, 3, 4);
    const b = Quaternion(f64).init(5, 6, 7, 8);
    var out: Quaternion(f64) = undefined;
    a.addAssign(&b, &out);

    try std.testing.expectEqual(a.add(b), out);
}

test "Quaternion mul" {
    const a = Quaternion(f64).init(1, 2, 3, 4);
    const b = Quaternion(f64).init(5, 6, 7, 8);
    const product = a.mul(b);

    try std.testing.expectApproxEqAbs(24, product.x, 1e-12);
    try std.testing.expectApproxEqAbs(48, product.y, 1e-12);
    try std.testing.expectApproxEqAbs(48, product.z, 1e-12);
    try std.testing.expectApproxEqAbs(-6, product.w, 1e-12);
}

test "Quaternion mulAssign" {
    const a = Quaternion(f64).init(1, 2, 3, 4);
    const b = Quaternion(f64).init(5, 6, 7, 8);
    var out: Quaternion(f64) = undefined;
    a.mulAssign(&b, &out);

    try std.testing.expectEqual(a.mul(b), out);
}

test "Quaternion conjugate" {
    const q = Quaternion(f64).init(1, -2, 3, 4);
    const c = q.conjugate();

    try std.testing.expectApproxEqAbs(-1, c.x, 1e-12);
    try std.testing.expectApproxEqAbs(2, c.y, 1e-12);
    try std.testing.expectApproxEqAbs(-3, c.z, 1e-12);
    try std.testing.expectApproxEqAbs(4, c.w, 1e-12);

    const norm_squared = q.mul(c);
    try std.testing.expectApproxEqAbs(0, norm_squared.x, 1e-12);
    try std.testing.expectApproxEqAbs(0, norm_squared.y, 1e-12);
    try std.testing.expectApproxEqAbs(0, norm_squared.z, 1e-12);
    try std.testing.expectApproxEqAbs(30, norm_squared.w, 1e-12);
}

test "Quaternion conjugateAssign" {
    const q = Quaternion(f64).init(1, -2, 3, 4);
    var out: Quaternion(f64) = undefined;
    q.conjugateAssign(&out);

    try std.testing.expectEqual(q.conjugate(), out);
}

test "Quaternion rotateVector z rotation" {
    const half_sqrt = std.math.sqrt(@as(f64, 0.5));
    const q = Quaternion(f64).init(0, 0, half_sqrt, half_sqrt);
    const v = @Vector(3, f64){ 1, 0, 0 };
    const rotated = q.rotateVector(v);

    try std.testing.expectApproxEqAbs(0, rotated[0], 1e-12);
    try std.testing.expectApproxEqAbs(1, rotated[1], 1e-12);
    try std.testing.expectApproxEqAbs(0, rotated[2], 1e-12);

    var assigned: @Vector(3, f64) = undefined;
    q.rotateVectorAssign(&v, &assigned);
    try std.testing.expectEqual(rotated, assigned);
}

test "Quaternion rotateVector matches matrix" {
    const q = Quaternion(f64).init(0.5, -0.5, 0.5, std.math.sqrt(@as(f64, 0.5))).normalized();
    const v = @Vector(3, f64){ 2, -3, 4 };

    const rotated = q.rotateVector(v);
    const matrix = q.toMat3();
    const expected = matrix.vecMul(&v);

    inline for (0..3) |idx| {
        try std.testing.expectApproxEqAbs(expected[idx], rotated[idx], 1e-12);
    }
}

test "Quaternion dot identity" {
    const q = Quaternion(f64).identity();
    const omega = @Vector(3, f64){ 2, -4, 6 };
    const q_dot = q.dot(omega);

    try std.testing.expectApproxEqAbs(1, q_dot.x, 1e-12);
    try std.testing.expectApproxEqAbs(-2, q_dot.y, 1e-12);
    try std.testing.expectApproxEqAbs(3, q_dot.z, 1e-12);
    try std.testing.expectApproxEqAbs(0, q_dot.w, 1e-12);
}

test "Quaternion dot" {
    const q = Quaternion(f64).init(1, 2, 3, 4);
    const omega = @Vector(3, f64){ 5, 6, 7 };
    const q_dot = q.dot(omega);

    try std.testing.expectApproxEqAbs(8, q_dot.x, 1e-12);
    try std.testing.expectApproxEqAbs(16, q_dot.y, 1e-12);
    try std.testing.expectApproxEqAbs(12, q_dot.z, 1e-12);
    try std.testing.expectApproxEqAbs(-19, q_dot.w, 1e-12);

    var assigned: Quaternion(f64) = undefined;
    q.dotAssign(&omega, &assigned);
    try std.testing.expectEqual(q_dot, assigned);
}

test "Quaternion toMat3 z rotation" {
    const half_sqrt = std.math.sqrt(@as(f64, 0.5));
    const q = Quaternion(f64).init(0, 0, half_sqrt, half_sqrt);
    const m = q.toMat3();

    var assigned: Mat(f64, 3, 3) = undefined;
    q.toMat3Assign(&assigned);
    try std.testing.expectEqual(m, assigned);

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

    var assigned: Quaternion(f64) = undefined;
    Quaternion(f64).fromMat3Assign(&matrix, &assigned);
    try std.testing.expectEqual(round_tripped, assigned);

    inline for (0..3) |col| {
        inline for (0..3) |row| {
            try std.testing.expectApproxEqAbs(matrix.data[col][row], round_tripped_matrix.data[col][row], 1e-12);
        }
    }
}

test "Mat init" {
    const data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const a = Mat(f32, 2, 3).init(data);
    try std.testing.expectEqual(a.data[0], @Vector(2, f32){ 1, 4 });
    try std.testing.expectEqual(a.data[1], @Vector(2, f32){ 2, 5 });
    try std.testing.expectEqual(a.data[2], @Vector(2, f32){ 3, 6 });

    var assigned: Mat(f32, 2, 3) = undefined;
    Mat(f32, 2, 3).initAssign(&data, &assigned);
    try std.testing.expectEqual(a, assigned);
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

    const c = a.matMul(&b);

    var assigned: Mat(f32, 2, 2) = undefined;
    a.matMulAssign(&b, &assigned);

    const expected = Mat(f32, 2, 2).init(.{
        22, 28,
        49, 64,
    });
    try std.testing.expectEqual(c, expected);
    try std.testing.expectEqual(c, assigned);
}

test "Mat vec_mul variants" {
    const a = Mat(f64, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    const b = @Vector(3, f64){ 7, 8, 9 };

    const result = a.vecMul(&b);
    var assigned: @Vector(2, f64) = undefined;
    a.vecMulAssign(&b, &assigned);

    try std.testing.expectEqual(@Vector(2, f64){ 50, 122 }, result);
    try std.testing.expectEqual(result, assigned);
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

    const c = a.matAdd(&b);

    const expected = Mat(f32, 2, 3).init(.{
        7, 7, 7,
        7, 7, 7,
    });
    try std.testing.expectEqual(expected, c);
}

test "Mat mat_add_assign" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    const b = Mat(f32, 2, 3).init(.{
        6, 5, 4,
        3, 2, 1,
    });

    var out: Mat(f32, 2, 3) = undefined;
    a.matAddAssign(&b, &out);

    const expected = Mat(f32, 2, 3).init(.{
        7, 7, 7,
        7, 7, 7,
    });
    try std.testing.expectEqual(expected, out);
}

test "Mat mat_neg" {
    const a = Mat(f32, 2, 3).init(.{
        1,  -2, 3,
        -4, 5,  -6,
    });

    const out = a.matNeg();

    const expected = Mat(f32, 2, 3).init(.{
        -1, 2,  -3,
        4,  -5, 6,
    });
    try std.testing.expectEqual(expected, out);
}

test "Mat mat_neg_assign" {
    const a = Mat(f32, 2, 3).init(.{
        1,  -2, 3,
        -4, 5,  -6,
    });

    var out: Mat(f32, 2, 3) = undefined;
    a.matNegAssign(&out);

    const expected = Mat(f32, 2, 3).init(.{
        -1, 2,  -3,
        4,  -5, 6,
    });
    try std.testing.expectEqual(expected, out);
}

test "Mat transpose" {
    const a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    const transposed = a.transpose();

    var assigned: Mat(f32, 3, 2) = undefined;
    a.transposeAssign(&assigned);

    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, transposed.data[0]);
    try std.testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, transposed.data[1]);
    try std.testing.expectEqual(transposed, assigned);
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

    try std.testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, a.getRow(1));

    a.setRow(0, @Vector(3, f32){ 7, 8, 9 });
    try std.testing.expectEqual(@Vector(3, f32){ 7, 8, 9 }, a.getRow(0));
    try std.testing.expectEqual(@Vector(2, f32){ 7, 4 }, a.data[0]);
}

test "Mat get and set col" {
    var a = Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });

    try std.testing.expectEqual(@Vector(2, f32){ 2, 5 }, a.getCol(1));

    a.setCol(1, @Vector(2, f32){ 8, 9 });
    try std.testing.expectEqual(@Vector(2, f32){ 8, 9 }, a.getCol(1));
    try std.testing.expectEqual(@as(f32, 8), a.get(0, 1));
}

test "Mat get and set block" {
    var a = Mat(f32, 3, 4).init(.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });

    const block = a.getBlock(2, 2, 1, 1);
    const expected = Mat(f32, 2, 2).init(.{
        6,  7,
        10, 11,
    });
    try std.testing.expectEqual(expected, block);

    const replacement = Mat(f32, 2, 2).init(.{
        20, 21,
        22, 23,
    });
    a.setBlock(0, 2, &replacement);

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
    try a.inverseAssign(&inv);

    try std.testing.expectApproxEqAbs(0.6, inv.data[0][0], 1e-10);
    try std.testing.expectApproxEqAbs(-0.2, inv.data[0][1], 1e-10);
    try std.testing.expectApproxEqAbs(-0.7, inv.data[1][0], 1e-10);
    try std.testing.expectApproxEqAbs(0.4, inv.data[1][1], 1e-10);
}

test "Mat inverse" {
    const a = Mat(f64, 2, 2).init(.{
        4, 7,
        2, 6,
    });

    const inv = try a.inverse();

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
    try a.inverseAssign(&inv);

    const product = a.matMul(&inv);

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
    try std.testing.expectError(error.SingularMatrix, a.inverseAssign(&inv));
    try std.testing.expectError(error.SingularMatrix, a.inverse());
}

test "Mat solve_lu" {
    const a = Mat(f64, 3, 3).init(.{
        3,  2,   -1,
        2,  -2,  4,
        -1, 0.5, -1,
    });
    const b = @Vector(3, f64){ 1, -2, 0 };

    const x = try a.solveLu(&b);

    var assigned: @Vector(3, f64) = undefined;
    try a.solveLuAssign(&b, &assigned);

    const expected = @Vector(3, f64){ 1, -2, -2 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
        try std.testing.expectApproxEqAbs(expected[i], assigned[i], 1e-10);
    }
}

test "Mat solve_lu singular" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, 4,
    });
    const b = @Vector(2, f64){ 1, 2 };

    var out: @Vector(2, f64) = undefined;
    try std.testing.expectError(error.SingularMatrix, a.solveLuAssign(&b, &out));
    try std.testing.expectError(error.SingularMatrix, a.solveLu(&b));
}

test "Mat solve_ldlt" {
    const a = Mat(f64, 3, 3).init(.{
        4, 1, 1,
        1, 3, 0,
        1, 0, 2,
    });
    const b = @Vector(3, f64){ 9, 7, 7 };

    const x = try a.solveLdlt(&b);

    var assigned: @Vector(3, f64) = undefined;
    try a.solveLdltAssign(&b, &assigned);

    const expected = @Vector(3, f64){ 1, 2, 3 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
        try std.testing.expectApproxEqAbs(expected[i], assigned[i], 1e-10);
    }
}

test "Mat solve_ldlt indefinite" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, -3,
    });
    const b = @Vector(2, f64){ 5, -4 };

    const x = try a.solveLdlt(&b);

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

    var out: @Vector(2, f64) = undefined;
    try std.testing.expectError(error.SingularMatrix, a.solveLdltAssign(&b, &out));
    try std.testing.expectError(error.SingularMatrix, a.solveLdlt(&b));
}

test "Mat solve_cholesky" {
    const a = Mat(f64, 3, 3).init(.{
        4, 1, 1,
        1, 3, 0,
        1, 0, 2,
    });
    const b = @Vector(3, f64){ 9, 7, 7 };

    const x = try a.solveCholesky(&b);

    var assigned: @Vector(3, f64) = undefined;
    try a.solveCholeskyAssign(&b, &assigned);

    const expected = @Vector(3, f64){ 1, 2, 3 };
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x[i], 1e-10);
        try std.testing.expectApproxEqAbs(expected[i], assigned[i], 1e-10);
    }
}

test "Mat solve_cholesky not positive definite" {
    const a = Mat(f64, 2, 2).init(.{
        1, 2,
        2, 1,
    });
    const b = @Vector(2, f64){ 1, 1 };

    var out: @Vector(2, f64) = undefined;
    try std.testing.expectError(error.NotPositiveDefinite, a.solveCholeskyAssign(&b, &out));
    try std.testing.expectError(error.NotPositiveDefinite, a.solveCholesky(&b));
}
