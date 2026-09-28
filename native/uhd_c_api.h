// Minimal declarations of the UHD C ABI used by the native TX player.
//
// Why not the C++ API directly?
//   UHD's public C++ headers (multi_usrp.hpp et al.) include Boost headers
//   (boost/core/noncopyable.hpp, boost/operators.hpp, ...).  The Windows UHD
//   runtime installer ships uhd.dll/uhd.lib + the public headers but *not*
//   Boost, so a plain MSVC build of the C++ headers cannot find <boost/...>.
//   UHD exports a stable, complete C ABI (uhd_usrp_*, uhd_tx_streamer_*,
//   uhd_tx_metadata_*) that drives the very same multi_usrp/tx_streamer
//   objects underneath.  We therefore declare only those prototypes here and
//   link against the same uhd.lib — no Boost needed, same streamer.
//
// The structs below mirror the exact layout of the public C headers in
// C:\Program Files\UHD\include (uhd/usrp/usrp.h, uhd/types/*.h).  The C
// handles are opaque pointers; we never dereference them.
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

// uhd_error is a C enum; int has the same size/ABI.
typedef int uhd_error;
static const uhd_error UHD_ERROR_NONE = 0;

typedef struct uhd_usrp* uhd_usrp_handle;
typedef struct uhd_tx_streamer* uhd_tx_streamer_handle;
typedef struct uhd_tx_metadata_t* uhd_tx_metadata_handle;
typedef struct uhd_async_metadata_t* uhd_async_metadata_handle;
typedef struct uhd_meta_range_t* uhd_meta_range_handle;

//! uhd::stream_args_t (uhd/usrp/usrp.h)
typedef struct {
    char* cpu_format;
    char* otw_format;
    char* args;
    size_t* channel_list;
    int n_channels;
} uhd_stream_args_t;

//! uhd::tune_request_t (uhd/types/tune_request.h)
typedef struct {
    double target_freq;
    int rf_freq_policy;   // 65 = AUTO, 77 = MANUAL, 78 = NONE
    double rf_freq;
    int dsp_freq_policy;
    double dsp_freq;
    char* args;
} uhd_tune_request_t;

//! uhd::tune_result_t (uhd/types/tune_result.h)
typedef struct {
    double clipped_rf_freq;
    double target_rf_freq;
    double actual_rf_freq;
    double target_dsp_freq;
    double actual_dsp_freq;
} uhd_tune_result_t;

// --- USRP handle / configuration -------------------------------------
uhd_error uhd_usrp_make(uhd_usrp_handle* h, const char* args);
uhd_error uhd_usrp_free(uhd_usrp_handle* h);
uhd_error uhd_usrp_last_error(uhd_usrp_handle h, char* error_out, size_t strbuffer_len);
uhd_error uhd_usrp_set_clock_source(uhd_usrp_handle h, const char* clock_source, size_t mboard);
uhd_error uhd_usrp_set_tx_rate(uhd_usrp_handle h, double rate, size_t chan);
uhd_error uhd_usrp_get_tx_rate(uhd_usrp_handle h, size_t chan, double* rate_out);
uhd_error uhd_usrp_set_tx_freq(uhd_usrp_handle h, uhd_tune_request_t* tune_request,
                               size_t chan, uhd_tune_result_t* tune_result);
uhd_error uhd_usrp_get_tx_freq(uhd_usrp_handle h, size_t chan, double* freq_out);
uhd_error uhd_usrp_set_tx_gain(uhd_usrp_handle h, double gain, size_t chan, const char* gain_name);
uhd_error uhd_usrp_get_tx_gain(uhd_usrp_handle h, size_t chan, const char* gain_name, double* gain_out);
uhd_error uhd_usrp_get_tx_gain_range(uhd_usrp_handle h, const char* name, size_t chan,
                                     uhd_meta_range_handle gain_range_out);
uhd_error uhd_usrp_set_tx_antenna(uhd_usrp_handle h, const char* ant, size_t chan);
uhd_error uhd_usrp_get_tx_antenna(uhd_usrp_handle h, size_t chan, char* ant_out, size_t strbuffer_len);
uhd_error uhd_usrp_set_tx_bandwidth(uhd_usrp_handle h, double bandwidth, size_t chan);
uhd_error uhd_usrp_get_tx_bandwidth(uhd_usrp_handle h, size_t chan, double* bandwidth_out);
uhd_error uhd_usrp_get_tx_num_channels(uhd_usrp_handle h, size_t* num_channels_out);
uhd_error uhd_usrp_get_tx_stream(uhd_usrp_handle h, uhd_stream_args_t* stream_args,
                                 uhd_tx_streamer_handle h_out);

// --- meta ranges ------------------------------------------------------
uhd_error uhd_meta_range_make(uhd_meta_range_handle* h);
uhd_error uhd_meta_range_free(uhd_meta_range_handle* h);
uhd_error uhd_meta_range_start(uhd_meta_range_handle h, double* start_out);
uhd_error uhd_meta_range_stop(uhd_meta_range_handle h, double* stop_out);
uhd_error uhd_meta_range_clip(uhd_meta_range_handle h, double value, bool clip_step,
                              double* result_out);

// --- TX streamer ------------------------------------------------------
uhd_error uhd_tx_streamer_make(uhd_tx_streamer_handle* h);
uhd_error uhd_tx_streamer_free(uhd_tx_streamer_handle* h);
uhd_error uhd_tx_streamer_max_num_samps(uhd_tx_streamer_handle h, size_t* max_num_samps_out);
uhd_error uhd_tx_streamer_send(uhd_tx_streamer_handle h, const void** buffs,
                               size_t samps_per_buff, uhd_tx_metadata_handle* md,
                               double timeout, size_t* items_sent);
uhd_error uhd_tx_streamer_recv_async_msg(uhd_tx_streamer_handle h,
                                         uhd_async_metadata_handle* md,
                                         double timeout, bool* valid);
uhd_error uhd_tx_streamer_last_error(uhd_tx_streamer_handle h, char* error_out, size_t strbuffer_len);

// --- TX metadata ------------------------------------------------------
uhd_error uhd_tx_metadata_make(uhd_tx_metadata_handle* handle, bool has_time_spec,
                               int64_t full_secs, double frac_secs,
                               bool start_of_burst, bool end_of_burst);
uhd_error uhd_tx_metadata_free(uhd_tx_metadata_handle* handle);

// --- async metadata ---------------------------------------------------
uhd_error uhd_async_metadata_make(uhd_async_metadata_handle* handle);
uhd_error uhd_async_metadata_free(uhd_async_metadata_handle* handle);
uhd_error uhd_async_metadata_event_code(uhd_async_metadata_handle h, int* event_code_out);

// --- global error string ---------------------------------------------
uhd_error uhd_get_last_error(char* error_out, size_t strbuffer_len);

}  // extern "C"

// Event-code bits (uhd/types/metadata.h).
enum {
    UHD_EVENT_BURST_ACK = 0x1,
    UHD_EVENT_UNDERFLOW = 0x2,
    UHD_EVENT_SEQ_ERROR = 0x4,
    UHD_EVENT_TIME_ERROR = 0x8,
    UHD_EVENT_UNDERFLOW_IN_PACKET = 0x10,
    UHD_EVENT_SEQ_ERROR_IN_BURST = 0x20
};
