#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fhp_plane { const uint8_t *data; size_t bytes; uint32_t width, height, stride; };

/* Thread-local description of the most recent failure. */
const char *fhg_error(void);

/* Masked native-camera area resize. Input planes are in trained model order;
 * each packed support array contains width*height zero or one bytes. */
int fhg_prepare_handspotter(const struct fhp_plane source[4], const uint8_t *support[4],
                            uint8_t output[81920]);

#ifdef __cplusplus
}
#endif
