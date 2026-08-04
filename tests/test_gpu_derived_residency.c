/* Synthetic coverage for Q8 aligned artifact residency.
 * No GGUF is opened: a small temporary file backs the raw lazy fallback. */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
            rc = 1;                                                     \
            goto cleanup;                                               \
        }                                                               \
    } while (0)

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_derived_residency: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices\n");
        return 0;
    }

    const uint64_t in_dim = 1024u;
    const uint64_t out_dim = 128u;
    const uint64_t raw_bytes = out_dim * (in_dim / 32u) * 34u;
    const uint64_t exact_offset = 4096u;
    const uint64_t disabled_offset = exact_offset + raw_bytes + 4096u;
    const uint64_t partial_offset = disabled_offset + raw_bytes + 4096u;
    const size_t model_size = 2u * 1024u * 1024u;
    const char *cache_label = "tensor:blk.0.ffn_gate_shexp.weight";

    int rc = 0;
    int fd = -1;
    void *model_map = MAP_FAILED;
    char path[] = "/tmp/ds4-q8-derived-test-XXXXXX";

    fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    CHECK(ftruncate(fd, (off_t)model_size) == 0, "ftruncate");
    model_map = mmap(NULL, model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    CHECK(model_map != MAP_FAILED, "mmap");
    for (size_t i = 0; i < model_size; i++) {
        ((unsigned char *)model_map)[i] = (unsigned char)(i * 17u + 3u);
    }
    CHECK(msync(model_map, model_size, MS_SYNC) == 0, "msync");

    unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    CHECK(setenv("DS4_CUDA_Q8_F16_ALL", "1", 1) == 0,
          "enable deterministic Q8 to F16 preload");
    CHECK(setenv("DS4_CUDA_Q8_F16_CACHE_MB", "16", 1) == 0,
          "bound Q8 to F16 test cache");
    CHECK(ds4_gpu_init(), "ds4_gpu_init");
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model_map), "set model fd/map");
    CHECK(ds4_gpu_test_add_q8_aligned_artifact(
              model_map, exact_offset, raw_bytes, in_dim, out_dim),
          "add exact Q8 aligned artifact");
    CHECK(ds4_gpu_test_derived_range_count() == 1, "one owned artifact");

    CHECK(ds4_gpu_model_range_replaced(model_map, exact_offset, raw_bytes) == 1,
          "exact Q8 source classified as replaced");
    CHECK(ds4_gpu_model_range_replaced(model_map, exact_offset, raw_bytes - 34u) == 0,
          "partial Q8 source is not replaced");
    CHECK(ds4_gpu_model_range_replaced(
              (const char *)model_map + 1, exact_offset, raw_bytes) == 0,
          "different model ownership is not replaced");

    CHECK(ds4_gpu_cache_model_range(model_map, model_size, exact_offset,
                                    raw_bytes, "exact Q8 raw preload"),
          "exact raw preload accepted as already replaced");
    CHECK(ds4_gpu_test_model_range_count() == 0,
          "exact aligned artifact skips initial raw residency");
    CHECK(ds4_gpu_cache_q8_f16_range(model_map, model_size, exact_offset,
                                     raw_bytes, in_dim, out_dim, cache_label),
          "exact Q8 to F16 preload skipped successfully");
    CHECK(ds4_gpu_test_q8_f16_range_count() == 0,
          "exact aligned artifact skips Q8 to F16 preload");

    const void *raw_fallback = ds4_gpu_test_resolve_raw_range(
        model_map, exact_offset, raw_bytes);
    CHECK(raw_fallback != NULL, "raw fallback resolves lazily");
    CHECK(ds4_gpu_test_model_range_count() == 1,
          "raw range appears only after lazy fallback");
    unsigned char got = 0;
    CHECK(cudaMemcpy(&got, raw_fallback, 1, cudaMemcpyDeviceToHost) == cudaSuccess,
          "read lazy raw fallback");
    CHECK(got == ((unsigned char *)model_map)[exact_offset],
          "lazy raw fallback preserves source bytes");

    CHECK(ds4_gpu_test_add_q8_aligned_artifact(
              model_map, disabled_offset, raw_bytes, in_dim, out_dim),
          "add artifact for disabled-path test");
    CHECK(setenv("DS4_CUDA_Q8_NO_ALIGNED", "1", 1) == 0,
          "disable aligned Q8");
    CHECK(ds4_gpu_model_range_replaced(model_map, disabled_offset, raw_bytes) == 0,
          "disabled artifact does not replace raw source");
    CHECK(ds4_gpu_cache_model_range(model_map, model_size, disabled_offset,
                                    raw_bytes, "disabled Q8 raw preload"),
          "disabled path retains raw preload");
    CHECK(ds4_gpu_test_model_range_count() == 2,
          "disabled path adds raw residency");
    CHECK(ds4_gpu_cache_q8_f16_range(model_map, model_size, disabled_offset,
                                     raw_bytes, in_dim, out_dim, cache_label),
          "disabled path retains Q8 to F16 preload");
    CHECK(ds4_gpu_test_q8_f16_range_count() == 1,
          "disabled path creates Q8 to F16 residency");
    CHECK(unsetenv("DS4_CUDA_Q8_NO_ALIGNED") == 0, "restore aligned Q8");

    CHECK(ds4_gpu_test_add_q8_aligned_artifact(
              model_map, partial_offset, raw_bytes - 34u, in_dim, out_dim),
          "add incomplete source artifact");
    CHECK(ds4_gpu_model_range_replaced(model_map, partial_offset, raw_bytes) == 0,
          "incomplete artifact does not replace full source");
    CHECK(setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1) == 0,
          "disable all derived weights");
    CHECK(ds4_gpu_model_range_replaced(model_map, exact_offset, raw_bytes) == 0,
          "global derived disable retains raw behavior");
    CHECK(unsetenv("DS4_CUDA_NO_DERIVED_WEIGHTS") == 0,
          "restore derived weights");

cleanup:
    unsetenv("DS4_CUDA_Q8_NO_ALIGNED");
    unsetenv("DS4_CUDA_NO_DERIVED_WEIGHTS");
    unsetenv("DS4_CUDA_Q8_F16_ALL");
    unsetenv("DS4_CUDA_Q8_F16_CACHE_MB");
    ds4_gpu_cleanup();
    if (ds4_gpu_test_model_range_count() != 0 ||
        ds4_gpu_test_q8_f16_range_count() != 0 ||
        ds4_gpu_test_derived_range_count() != 0) {
        fprintf(stderr, "FAIL: cleanup left residency state\n");
        rc = 1;
    }
    ds4_gpu_cleanup();
    if (ds4_gpu_test_model_range_count() != 0 ||
        ds4_gpu_test_q8_f16_range_count() != 0 ||
        ds4_gpu_test_derived_range_count() != 0) {
        fprintf(stderr, "FAIL: repeated cleanup is not idempotent\n");
        rc = 1;
    }
    if (model_map != MAP_FAILED) (void)munmap(model_map, model_size);
    if (fd >= 0) (void)close(fd);
    (void)unlink(path);
    if (rc == 0) fprintf(stderr, "test_gpu_derived_residency PASS\n");
    return rc;
}
