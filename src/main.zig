const std = @import("std");
const zla = @import("zla");

pub fn main() !void {
    const mat_type = zla.Mat(f32, 2, 3).init(.{
        1, 2, 3,
        4, 5, 6,
    });
    std.debug.print("{any}\n", .{@typeInfo(@TypeOf(mat_type)).@"struct".fields[1]});
    std.debug.print("{any}\n", .{mat_type.rows});
    std.debug.print("{f}\n", .{mat_type});
}
