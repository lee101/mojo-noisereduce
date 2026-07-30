"""Spectral-gating kernels exposed through a C ABI."""

from std.algorithm import sync_parallelize
from std.math import exp, log10, sqrt
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime EPS = 2.220446049250313e-16
comptime TASKS = 16
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def task_count(rows: Int) -> Int:
    return min(rows, TASKS)


def stationary_threshold_row(
    magnitude: Ptr,
    means: Ptr,
    deviations: Ptr,
    thresholds: Ptr,
    row: Int,
    cols: Int,
    n_std: Float64,
):
    var offset = row * cols
    var vector_stop = cols - cols % W
    var max_magnitude = magnitude[offset]
    var max_start = 1
    if vector_stop > 0:
        var maxima = magnitude.load[width=W](offset)
        for col in range(W, vector_stop, W):
            maxima = max(
                maxima, magnitude.load[width=W](offset + col)
            )
        max_magnitude = maxima.reduce_max()
        max_start = vector_stop
    for col in range(max_start, cols):
        max_magnitude = max(max_magnitude, magnitude[offset + col])
    var max_db = 20.0 * log10(max_magnitude + EPS)
    var floor_db = max_db - 80.0
    var total = 0.0
    for col in range(0, vector_stop, W):
        var values = (
            20.0
            * log10(magnitude.load[width=W](offset + col) + EPS)
        )
        total += max(values, floor_db).reduce_add()
    for col in range(vector_stop, cols):
        var value = 20.0 * log10(magnitude[offset + col] + EPS)
        total += max(value, floor_db)
    var mean = total / Float64(cols)
    var squared = 0.0
    for col in range(0, vector_stop, W):
        var values = (
            20.0
            * log10(magnitude.load[width=W](offset + col) + EPS)
        )
        var deltas = max(values, floor_db) - mean
        squared += (deltas * deltas).reduce_add()
    for col in range(vector_stop, cols):
        var value = 20.0 * log10(magnitude[offset + col] + EPS)
        var delta = max(value, floor_db) - mean
        squared += delta * delta
    var deviation = sqrt(squared / Float64(cols))
    means[row] = mean
    deviations[row] = deviation
    thresholds[row] = mean + n_std * deviation


def stationary_threshold(
    magnitude: Ptr,
    means: Ptr,
    deviations: Ptr,
    thresholds: Ptr,
    rows: Int,
    cols: Int,
    n_std: Float64,
):
    @parameter
    def work(task: Int):
        var tasks = task_count(rows)
        var start = task * rows // tasks
        var stop = (task + 1) * rows // tasks
        for row in range(start, stop):
            stationary_threshold_row(
                magnitude,
                means,
                deviations,
                thresholds,
                row,
                cols,
                n_std,
            )

    if rows * cols < 65536:
        for row in range(rows):
            stationary_threshold_row(
                magnitude,
                means,
                deviations,
                thresholds,
                row,
                cols,
                n_std,
            )
    else:
        sync_parallelize[work](task_count(rows))


def stationary_raw_row(
    magnitude: Ptr,
    thresholds: Ptr,
    mask: Ptr,
    row: Int,
    cols: Int,
    prop_decrease: Float64,
):
    var offset = row * cols
    var vector_stop = cols - cols % W
    var max_magnitude = magnitude[offset]
    var max_start = 1
    if vector_stop > 0:
        var maxima = magnitude.load[width=W](offset)
        for col in range(W, vector_stop, W):
            maxima = max(
                maxima, magnitude.load[width=W](offset + col)
            )
        max_magnitude = maxima.reduce_max()
        max_start = vector_stop
    for col in range(max_start, cols):
        max_magnitude = max(max_magnitude, magnitude[offset + col])
    var max_db = 20.0 * log10(max_magnitude + EPS)
    var floor_db = max_db - 80.0
    var residual = 1.0 - prop_decrease
    if floor_db > thresholds[row]:
        for col in range(0, vector_stop, W):
            mask.store(
                offset + col, SIMD[DType.float64, W](1.0)
            )
        for col in range(vector_stop, cols):
            mask[offset + col] = 1.0
    else:
        for col in range(0, vector_stop, W):
            var values = (
                20.0
                * log10(
                    magnitude.load[width=W](offset + col) + EPS
                )
            )
            mask.store(
                offset + col,
                values.gt(thresholds[row]).select(
                    SIMD[DType.float64, W](1.0),
                    SIMD[DType.float64, W](residual),
                ),
            )
        for col in range(vector_stop, cols):
            var value = (
                20.0 * log10(magnitude[offset + col] + EPS)
            )
            mask[offset + col] = (
                1.0
                if value > thresholds[row]
                else residual
            )


def stationary_raw(
    magnitude: Ptr,
    thresholds: Ptr,
    mask: Ptr,
    rows: Int,
    cols: Int,
    prop_decrease: Float64,
):
    @parameter
    def work(task: Int):
        var tasks = task_count(rows)
        var start = task * rows // tasks
        var stop = (task + 1) * rows // tasks
        for row in range(start, stop):
            stationary_raw_row(
                magnitude, thresholds, mask, row, cols, prop_decrease
            )

    if rows * cols < 65536:
        for row in range(rows):
            stationary_raw_row(
                magnitude, thresholds, mask, row, cols, prop_decrease
            )
    else:
        sync_parallelize[work](task_count(rows))


def convolve_frequency_row(
    source: Ptr,
    destination: Ptr,
    freq_weights: Ptr,
    row: Int,
    rows: Int,
    cols: Int,
    freq_len: Int,
):
    var radius = freq_len // 2
    var vector_stop = cols - cols % W
    for col in range(0, vector_stop, W):
        var total = SIMD[DType.float64, W](0.0)
        for kernel_row in range(freq_len):
            var source_row = row + kernel_row - radius
            if source_row >= 0 and source_row < rows:
                total += (
                    source.load[width=W](source_row * cols + col)
                    * freq_weights[kernel_row]
                )
        destination.store(row * cols + col, total)
    for col in range(vector_stop, cols):
        var total = 0.0
        for kernel_row in range(freq_len):
            var source_row = row + kernel_row - radius
            if source_row >= 0 and source_row < rows:
                total += (
                    source[source_row * cols + col]
                    * freq_weights[kernel_row]
                )
        destination[row * cols + col] = total


def convolve_frequency(
    source: Ptr,
    destination: Ptr,
    freq_weights: Ptr,
    rows: Int,
    cols: Int,
    freq_len: Int,
):
    @parameter
    def work(task: Int):
        var tasks = task_count(rows)
        var start = task * rows // tasks
        var stop = (task + 1) * rows // tasks
        for row in range(start, stop):
            convolve_frequency_row(
                source,
                destination,
                freq_weights,
                row,
                rows,
                cols,
                freq_len,
            )

    if rows * cols * freq_len < 65536:
        for row in range(rows):
            convolve_frequency_row(
                source,
                destination,
                freq_weights,
                row,
                rows,
                cols,
                freq_len,
            )
    else:
        sync_parallelize[work](task_count(rows))


def convolve_time_row(
    source: Ptr,
    destination: Ptr,
    time_weights: Ptr,
    row: Int,
    cols: Int,
    time_len: Int,
    prop_decrease: Float64,
    blend: Bool,
):
    var radius = time_len // 2
    var offset = row * cols
    var residual = 1.0 - prop_decrease
    var interior_stop = cols - (time_len - radius - 1)
    var vector_stop = (
        radius + max(interior_stop - radius, 0) // W * W
    )
    for col in range(min(radius, cols)):
        var total = 0.0
        for kernel_col in range(time_len):
            var source_col = col + kernel_col - radius
            if source_col >= 0 and source_col < cols:
                total += (
                    source[offset + source_col]
                    * time_weights[kernel_col]
                )
        destination[offset + col] = (
            total * prop_decrease + residual if blend else total
        )
    for col in range(radius, vector_stop, W):
        var total = SIMD[DType.float64, W](0.0)
        for kernel_col in range(time_len):
            total += (
                source.load[width=W](
                    offset + col + kernel_col - radius
                )
                * time_weights[kernel_col]
            )
        destination.store(
            offset + col,
            (
                total * prop_decrease + residual
                if blend
                else total
            ),
        )
    for col in range(vector_stop, cols):
        var total = 0.0
        for kernel_col in range(time_len):
            var source_col = col + kernel_col - radius
            if source_col >= 0 and source_col < cols:
                total += (
                    source[offset + source_col]
                    * time_weights[kernel_col]
                )
        destination[offset + col] = (
            total * prop_decrease + residual if blend else total
        )


def convolve_time(
    source: Ptr,
    destination: Ptr,
    time_weights: Ptr,
    rows: Int,
    cols: Int,
    time_len: Int,
    prop_decrease: Float64,
    blend: Bool,
):
    @parameter
    def work(task: Int):
        var tasks = task_count(rows)
        var start = task * rows // tasks
        var stop = (task + 1) * rows // tasks
        for row in range(start, stop):
            convolve_time_row(
                source,
                destination,
                time_weights,
                row,
                cols,
                time_len,
                prop_decrease,
                blend,
            )

    if rows * cols * time_len < 65536:
        for row in range(rows):
            convolve_time_row(
                source,
                destination,
                time_weights,
                row,
                cols,
                time_len,
                prop_decrease,
                blend,
            )
    else:
        sync_parallelize[work](task_count(rows))


def nonstationary_raw_row(
    magnitude: Ptr,
    mask: Ptr,
    smooth: Ptr,
    row: Int,
    cols: Int,
    coefficient: Float64,
    threshold: Float64,
    slope: Float64,
):
    var offset = row * cols
    smooth[offset] = magnitude[offset]
    for col in range(1, cols):
        smooth[offset + col] = (
            coefficient * magnitude[offset + col]
            + (1.0 - coefficient) * smooth[offset + col - 1]
        )
    for reverse_col in range(cols - 1):
        var col = cols - 2 - reverse_col
        smooth[offset + col] = (
            coefficient * smooth[offset + col]
            + (1.0 - coefficient) * smooth[offset + col + 1]
        )
    var vector_stop = cols - cols % W
    for col in range(0, vector_stop, W):
        var smooth_values = smooth.load[width=W](offset + col)
        var above = (
            (
                magnitude.load[width=W](offset + col)
                - smooth_values
            )
            / smooth_values
        )
        mask.store(
            offset + col,
            1.0 / (1.0 + exp(-(above - threshold) * slope)),
        )
    for col in range(vector_stop, cols):
        var above = (
            (magnitude[offset + col] - smooth[offset + col])
            / smooth[offset + col]
        )
        mask[offset + col] = (
            1.0 / (1.0 + exp(-(above - threshold) * slope))
        )


def nonstationary_raw(
    magnitude: Ptr,
    mask: Ptr,
    smooth: Ptr,
    rows: Int,
    cols: Int,
    coefficient: Float64,
    threshold: Float64,
    slope: Float64,
):
    @parameter
    def work(task: Int):
        var tasks = task_count(rows)
        var start = task * rows // tasks
        var stop = (task + 1) * rows // tasks
        for row in range(start, stop):
            nonstationary_raw_row(
                magnitude,
                mask,
                smooth,
                row,
                cols,
                coefficient,
                threshold,
                slope,
            )

    if rows * cols < 65536:
        for row in range(rows):
            nonstationary_raw_row(
                magnitude,
                mask,
                smooth,
                row,
                cols,
                coefficient,
                threshold,
                slope,
            )
    else:
        sync_parallelize[work](task_count(rows))


@export("mnr_stationary_threshold")
def mnr_stationary_threshold(
    magnitude_addr: Int,
    rows: Int,
    cols: Int,
    n_std: Float64,
    means_addr: Int,
    deviations_addr: Int,
    thresholds_addr: Int,
) abi("C"):
    if rows > 0 and cols > 0:
        stationary_threshold(
            p(magnitude_addr),
            p(means_addr),
            p(deviations_addr),
            p(thresholds_addr),
            rows,
            cols,
            n_std,
        )


@export("mnr_stationary_mask")
def mnr_stationary_mask(
    magnitude_addr: Int,
    thresholds_addr: Int,
    rows: Int,
    cols: Int,
    prop_decrease: Float64,
    freq_weights_addr: Int,
    freq_len: Int,
    time_weights_addr: Int,
    time_len: Int,
    mask_addr: Int,
    work_addr: Int,
) abi("C"):
    if rows <= 0 or cols <= 0:
        return
    stationary_raw(
        p(magnitude_addr),
        p(thresholds_addr),
        p(mask_addr),
        rows,
        cols,
        prop_decrease,
    )
    convolve_frequency(
        p(mask_addr),
        p(work_addr),
        p(freq_weights_addr),
        rows,
        cols,
        freq_len,
    )
    convolve_time(
        p(work_addr),
        p(mask_addr),
        p(time_weights_addr),
        rows,
        cols,
        time_len,
        1.0,
        False,
    )


@export("mnr_nonstationary_mask")
def mnr_nonstationary_mask(
    magnitude_addr: Int,
    rows: Int,
    cols: Int,
    coefficient: Float64,
    threshold: Float64,
    slope: Float64,
    prop_decrease: Float64,
    freq_weights_addr: Int,
    freq_len: Int,
    time_weights_addr: Int,
    time_len: Int,
    mask_addr: Int,
    work_addr: Int,
) abi("C"):
    if rows <= 0 or cols <= 0:
        return
    nonstationary_raw(
        p(magnitude_addr),
        p(mask_addr),
        p(work_addr),
        rows,
        cols,
        coefficient,
        threshold,
        slope,
    )
    convolve_frequency(
        p(mask_addr),
        p(work_addr),
        p(freq_weights_addr),
        rows,
        cols,
        freq_len,
    )
    convolve_time(
        p(work_addr),
        p(mask_addr),
        p(time_weights_addr),
        rows,
        cols,
        time_len,
        prop_decrease,
        True,
    )
