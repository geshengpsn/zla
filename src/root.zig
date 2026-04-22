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
