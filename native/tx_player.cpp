// gnss_sim_tx — native UHD TX player for GNSS_Sim.
//
// Streams an interleaved cs16 (int16 I/Q) file to a USRP TX channel from a
// tight C++ loop (no Python/GIL per-block overhead).  This is the low-overhead
// path used to sustain ~25 Msps without the TX underflows seen with the Python
// UHD sink.
//
// Build with native/build.bat (MSVC + UHD import library); see uhd_c_api.h for
// why the program links against UHD's stable C ABI instead of the C++ headers.
//
// Safety: this program is TX-only.  It never opens an RX stream and never maps
// an RX channel, so TX and RX can never share a channel here.  Gain is clamped
// to the device TX gain range.  Use only inside a shielded/attenuated path.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "uhd_c_api.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop(false);

void on_signal(int) {
    g_stop.store(true);
}

struct Options {
    std::string file;
    double rate = 0.0;
    double freq = 0.0;
    double gain = 18.0;
    std::string antenna = "TX/RX";
    double bandwidth = 0.0;
    std::string args = "type=b200";
    std::string clock_source = "internal";
    std::string format = "cs16";
    size_t channel = 0;
    bool loop = false;
    double seconds = 0.0;
    size_t frame_size = 16384;
    size_t num_frames = 32;
    std::string stream_args;
    bool dry_run = false;
};

void print_usage() {
    std::fprintf(stderr,
        "gnss_sim_tx — native UHD cs16 TX player\n"
        "  --file PATH        interleaved int16 I/Q file (required)\n"
        "  --rate HZ          TX sample rate (required)\n"
        "  --freq HZ          TX centre frequency (required)\n"
        "  --gain DB          TX gain, clamped to device range (default 18)\n"
        "  --antenna NAME     TX antenna/port (default TX/RX)\n"
        "  --bandwidth HZ     analog TX bandwidth (0 = UHD default)\n"
        "  --args STR         UHD device args (default type=b200)\n"
        "  --channel N        TX channel (default 0)\n"
        "  --clock-source S   clock source (default internal)\n"
        "  --loop             loop the file until stopped\n"
        "  --seconds S        stop after S seconds of stream (0 = whole file/\n"
        "                     forever when --loop)\n"
        "  --frame-size N     requested samples per UHD frame (clamped to the\n"
        "                     streamer max; default 16384)\n"
        "  --num-frames N     bounded read-ahead frames (back-pressure, default 32)\n"
        "  --stream-args STR  UHD stream args (default num_send_frames=N)\n"
        "  --dry-run          open/configure the device, send nothing, exit\n"
        "  --format cs16      sample format (only cs16 is implemented)\n"
        "  --help             show this help\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string key = a;
        std::string val;
        bool has_val = false;
        const size_t eq = a.find('=');
        if (eq != std::string::npos) {
            key = a.substr(0, eq);
            val = a.substr(eq + 1);
            has_val = true;
        }
        auto need = [&](const char* name) -> std::string {
            if (has_val) return val;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (key == "--help" || key == "-h") {
            print_usage();
            std::exit(0);
        } else if (key == "--file") {
            o.file = need("--file");
        } else if (key == "--rate") {
            o.rate = std::atof(need("--rate").c_str());
        } else if (key == "--freq") {
            o.freq = std::atof(need("--freq").c_str());
        } else if (key == "--gain") {
            o.gain = std::atof(need("--gain").c_str());
        } else if (key == "--antenna") {
            o.antenna = need("--antenna");
        } else if (key == "--bandwidth") {
            o.bandwidth = std::atof(need("--bandwidth").c_str());
        } else if (key == "--args") {
            o.args = need("--args");
        } else if (key == "--channel") {
            o.channel = static_cast<size_t>(std::atoi(need("--channel").c_str()));
        } else if (key == "--clock-source") {
            o.clock_source = need("--clock-source");
        } else if (key == "--format") {
            o.format = need("--format");
        } else if (key == "--loop") {
            o.loop = true;
        } else if (key == "--dry-run") {
            o.dry_run = true;
        } else if (key == "--seconds") {
            o.seconds = std::atof(need("--seconds").c_str());
        } else if (key == "--frame-size") {
            o.frame_size = static_cast<size_t>(std::atoll(need("--frame-size").c_str()));
        } else if (key == "--num-frames") {
            o.num_frames = static_cast<size_t>(std::atoll(need("--num-frames").c_str()));
        } else if (key == "--stream-args") {
            o.stream_args = need("--stream-args");
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage();
            std::exit(2);
        }
    }
    if (o.file.empty() || o.rate <= 0.0 || o.freq <= 0.0) {
        std::fprintf(stderr, "error: --file, --rate and --freq are required\n");
        return false;
    }
    if (o.frame_size == 0) o.frame_size = 16384;
    if (o.num_frames == 0) o.num_frames = 32;
    return true;
}

std::string last_uhd_error() {
    char buf[1024] = {0};
    uhd_get_last_error(buf, sizeof(buf));
    return std::string(buf);
}

#define UHD_CHECK(expr, what)                                                   \
    do {                                                                        \
        uhd_error _e = (expr);                                                  \
        if (_e != UHD_ERROR_NONE) {                                             \
            std::fprintf(stderr, "UHD error in %s (code %d): %s\n", what,       \
                         static_cast<int>(_e), last_uhd_error().c_str());       \
            return 3;                                                           \
        }                                                                       \
    } while (0)

void boost_current_thread() {
    // Best-effort: raise the whole process so neither the reader nor the send
    // loop is descheduled mid-stream.  Per-thread TIME_CRITICAL priority used to
    // starve the reader thread (worse than one high-priority process).
    ::SetPriorityClass(::GetCurrentProcess(), HIGH_PRIORITY_CLASS);
}

bool query_gain_range(uhd_usrp_handle usrp, size_t chan, double* lo, double* hi) {
    // UHD's ALL_GAINS sentinel is "all"; fall back to "" for other bindings.
    const char* names[2] = {"all", ""};
    for (const char* name : names) {
        uhd_meta_range_handle rng = nullptr;
        if (uhd_meta_range_make(&rng) != UHD_ERROR_NONE) continue;
        bool ok = uhd_usrp_get_tx_gain_range(usrp, name, chan, rng) == UHD_ERROR_NONE;
        if (ok) {
            double a = 0.0, b = 0.0;
            ok = (uhd_meta_range_start(rng, &a) == UHD_ERROR_NONE)
                 && (uhd_meta_range_stop(rng, &b) == UHD_ERROR_NONE);
            if (ok) {
                *lo = a;
                *hi = b;
            }
        }
        uhd_meta_range_free(&rng);
        if (ok && *hi > *lo) return true;
    }
    return false;
}

struct Frame {
    std::vector<int16_t> samples;  // interleaved I/Q
    size_t nsamp = 0;              // complex samples
};

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
#ifdef SIGBREAK
    std::signal(SIGBREAK, on_signal);
#endif

    Options o;
    if (!parse_args(argc, argv, o)) return 2;
    if (o.format != "cs16") {
        std::fprintf(stderr, "error: only --format cs16 is implemented (got %s)\n",
                     o.format.c_str());
        return 2;
    }

    // Validate the file up front so a typo never opens/keys the radio.
    std::ifstream probe(o.file, std::ios::binary | std::ios::ate);
    if (!probe.is_open()) {
        std::fprintf(stderr, "error: cannot open file %s\n", o.file.c_str());
        return 2;
    }
    const std::streamoff file_bytes = probe.tellg();
    probe.close();
    const uint64_t file_samps = static_cast<uint64_t>(file_bytes) / 4;  // cs16 = 4 B
    if (file_samps == 0) {
        std::fprintf(stderr, "error: file %s has no cs16 samples\n", o.file.c_str());
        return 2;
    }

    // ---- open and configure the device (TX only) --------------------
    uhd_usrp_handle usrp = nullptr;
    UHD_CHECK(uhd_usrp_make(&usrp, o.args.c_str()), "uhd_usrp_make");

    if (!o.clock_source.empty())
        uhd_usrp_set_clock_source(usrp, o.clock_source.c_str(), 0);  // best effort

    UHD_CHECK(uhd_usrp_set_tx_rate(usrp, o.rate, o.channel), "set_tx_rate");
    uhd_tune_request_t tune;
    std::memset(&tune, 0, sizeof(tune));
    tune.target_freq = o.freq;
    tune.rf_freq_policy = 65;   // AUTO
    tune.dsp_freq_policy = 65;  // AUTO
    uhd_tune_result_t tune_res;
    std::memset(&tune_res, 0, sizeof(tune_res));
    UHD_CHECK(uhd_usrp_set_tx_freq(usrp, &tune, o.channel, &tune_res), "set_tx_freq");

    if (o.bandwidth > 0.0) {
        if (uhd_usrp_set_tx_bandwidth(usrp, o.bandwidth, o.channel) != UHD_ERROR_NONE)
            std::fprintf(stderr, "warning: set_tx_bandwidth failed (continuing)\n");
    }
    if (!o.antenna.empty()
        && uhd_usrp_set_tx_antenna(usrp, o.antenna.c_str(), o.channel) != UHD_ERROR_NONE)
        std::fprintf(stderr, "warning: set_tx_antenna(%s) failed (continuing)\n",
                     o.antenna.c_str());

    double gmin = 0.0, gmax = 0.0;
    double gain = o.gain;
    if (query_gain_range(usrp, o.channel, &gmin, &gmax)) {
        if (gain < gmin) {
            std::fprintf(stderr,
                "warning: requested gain %.2f dB below device range %.2f..%.2f — "
                "clamped to %.2f\n", gain, gmin, gmax, gmin);
            gain = gmin;
        } else if (gain > gmax) {
            std::fprintf(stderr,
                "warning: requested gain %.2f dB above device range %.2f..%.2f — "
                "clamped to %.2f\n", gain, gmin, gmax, gmax);
            gain = gmax;
        }
    }
    if (uhd_usrp_set_tx_gain(usrp, gain, o.channel, "all") != UHD_ERROR_NONE) {
        if (uhd_usrp_set_tx_gain(usrp, gain, o.channel, "") != UHD_ERROR_NONE)
            std::fprintf(stderr, "warning: set_tx_gain failed (continuing)\n");
    }

    // ---- TX streamer -------------------------------------------------
    size_t chan_list[1] = {o.channel};
    char cpu_fmt[] = "sc16";
    char otw_fmt[] = "sc16";
    std::string stream_args_str = o.stream_args;
    if (stream_args_str.empty())
        stream_args_str = "num_send_frames=" + std::to_string(o.num_frames);
    std::vector<char> sargs_buf(stream_args_str.begin(), stream_args_str.end());
    sargs_buf.push_back('\0');
    uhd_stream_args_t sargs;
    std::memset(&sargs, 0, sizeof(sargs));
    sargs.cpu_format = cpu_fmt;
    sargs.otw_format = otw_fmt;
    sargs.args = sargs_buf.data();
    sargs.channel_list = chan_list;
    sargs.n_channels = 1;

    uhd_tx_streamer_handle streamer = nullptr;
    UHD_CHECK(uhd_tx_streamer_make(&streamer), "uhd_tx_streamer_make");
    UHD_CHECK(uhd_usrp_get_tx_stream(usrp, &sargs, streamer), "get_tx_stream");

    size_t max_samps = 0;
    UHD_CHECK(uhd_tx_streamer_max_num_samps(streamer, &max_samps), "max_num_samps");
    if (max_samps == 0) max_samps = 2040;

    uhd_tx_metadata_handle md_start = nullptr, md_mid = nullptr, md_eob = nullptr;
    UHD_CHECK(uhd_tx_metadata_make(&md_start, false, 0, 0.0, true, false), "md_start");
    UHD_CHECK(uhd_tx_metadata_make(&md_mid, false, 0, 0.0, false, false), "md_mid");
    UHD_CHECK(uhd_tx_metadata_make(&md_eob, false, 0, 0.0, false, true), "md_eob");

    uhd_async_metadata_handle amd = nullptr;
    const bool have_amd = uhd_async_metadata_make(&amd) == UHD_ERROR_NONE;

    // frame_size is the application-level send chunk; UHD fragments it into
    // max_num_samps packets internally (B2xx has no set_tx_frame_size in UHD
    // 4.10 — the C++ API only exposes set_rx_spp).  Larger chunks mean fewer
    // send() calls and a deeper transport queue, which is what fights
    // underflow on Windows USB3.
    const size_t frame_samples = std::max<size_t>(1, o.frame_size);

    double actual_rate = 0.0, actual_freq = 0.0, actual_bw = 0.0;
    uhd_usrp_get_tx_rate(usrp, o.channel, &actual_rate);
    uhd_usrp_get_tx_freq(usrp, o.channel, &actual_freq);
    uhd_usrp_get_tx_bandwidth(usrp, o.channel, &actual_bw);

    std::fprintf(stderr,
        "[gnss_sim_tx] file=%s samps=%llu | %.3f Msps, %.3f MHz, gain %.2f dB, "
        "ant %s, bw %.3f MHz, ch %zu | frame %zu (max %zu), num_frames %zu, "
        "stream_args '%s', loop=%d, seconds=%.1f\n",
        o.file.c_str(), static_cast<unsigned long long>(file_samps),
        actual_rate / 1e6, actual_freq / 1e6, gain, o.antenna.c_str(),
        actual_bw / 1e6, o.channel, frame_samples, max_samps, o.num_frames,
        stream_args_str.c_str(), o.loop ? 1 : 0, o.seconds);
    std::fflush(stderr);

    if (o.dry_run) {
        std::fprintf(stderr, "[gnss_sim_tx] dry-run: device configured, no samples sent\n");
        if (have_amd) uhd_async_metadata_free(&amd);
        uhd_tx_metadata_free(&md_start);
        uhd_tx_metadata_free(&md_mid);
        uhd_tx_metadata_free(&md_eob);
        uhd_tx_streamer_free(&streamer);
        uhd_usrp_free(&usrp);
        return 0;
    }

    // ---- bounded read-ahead producer ---------------------------------
    std::mutex mtx;
    std::condition_variable cv_can_push, cv_can_pop;
    std::deque<std::shared_ptr<Frame>> queue;
    bool reader_done = false;
    const size_t queue_cap = std::max<size_t>(2, o.num_frames);

    std::thread reader([&]() {
        boost_current_thread();
        std::ifstream in(o.file, std::ios::binary);
        std::vector<int16_t> raw(frame_samples * 2);
        while (!g_stop.load()) {
            in.read(reinterpret_cast<char*>(raw.data()),
                    static_cast<std::streamsize>(frame_samples * 4));
            const std::streamsize got = in.gcount();
            size_t ncomp = static_cast<size_t>(got) / 4;
            if (ncomp == 0) {
                if (o.loop && !g_stop.load()) {
                    in.clear();
                    in.seekg(0);
                    continue;
                }
                break;
            }
            auto frame = std::make_shared<Frame>();
            frame->samples.assign(raw.begin(), raw.begin() + ncomp * 2);
            frame->nsamp = ncomp;
            std::unique_lock<std::mutex> lk(mtx);
            cv_can_push.wait(lk, [&]() { return g_stop.load() || queue.size() < queue_cap; });
            if (g_stop.load()) break;
            queue.push_back(frame);
            lk.unlock();
            cv_can_pop.notify_one();
        }
        std::unique_lock<std::mutex> lk(mtx);
        reader_done = true;
        lk.unlock();
        cv_can_pop.notify_all();
    });

    // ---- tight send loop ---------------------------------------------
    boost_current_thread();
    const uint64_t budget = (o.seconds > 0.0)
        ? static_cast<uint64_t>(std::llround(o.seconds * actual_rate)) : 0;
    uint64_t sent = 0;
    uint64_t short_sends = 0;
    uint64_t async_underflows = 0;
    uint64_t async_seq_errors = 0;
    bool first_send = true;
    bool send_error = false;

    const auto wall0 = std::chrono::steady_clock::now();
    auto last_status = wall0;
    uint64_t last_status_sent = 0;

    while (!g_stop.load() && (budget == 0 || sent < budget)) {
        std::shared_ptr<Frame> frame;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_can_pop.wait(lk, [&]() {
                return g_stop.load() || !queue.empty() || reader_done;
            });
            if (g_stop.load()) break;
            if (queue.empty()) {
                if (reader_done) break;
                continue;
            }
            frame = queue.front();
            queue.pop_front();
        }
        cv_can_push.notify_one();

        size_t off = 0;
        while (off < frame->nsamp && !g_stop.load()) {
            size_t want = frame->nsamp - off;
            if (budget != 0 && sent + want > budget) want = static_cast<size_t>(budget - sent);
            if (want == 0) break;
            const size_t n = want;
            const void* buffs[1] = {static_cast<const void*>(frame->samples.data() + off * 2)};
            uhd_tx_metadata_handle md = first_send ? md_start : md_mid;
            size_t items = 0;
            const uhd_error e = uhd_tx_streamer_send(streamer, buffs, n, &md, 1.0, &items);
            if (e != UHD_ERROR_NONE) {
                char err[1024] = {0};
                uhd_tx_streamer_last_error(streamer, err, sizeof(err));
                std::fprintf(stderr, "[gnss_sim_tx] send error (code %d): %s\n",
                             static_cast<int>(e), err);
                send_error = true;
                g_stop.store(true);
                break;
            }
            if (items < n) ++short_sends;
            sent += items;
            off += items;
            first_send = false;

            // Drain async TX events (underflow/sequence/time errors).
            if (have_amd) {
                while (true) {
                    bool valid = false;
                    if (uhd_tx_streamer_recv_async_msg(streamer, &amd, 0.0, &valid)
                            != UHD_ERROR_NONE || !valid)
                        break;
                    int code = 0;
                    if (uhd_async_metadata_event_code(amd, &code) != UHD_ERROR_NONE)
                        break;
                    if (code & (UHD_EVENT_UNDERFLOW | UHD_EVENT_UNDERFLOW_IN_PACKET))
                        ++async_underflows;
                    if (code & (UHD_EVENT_SEQ_ERROR | UHD_EVENT_SEQ_ERROR_IN_BURST))
                        ++async_seq_errors;
                }
            }
        }
        if (send_error) break;

        const auto now = std::chrono::steady_clock::now();
        if (now - last_status >= std::chrono::seconds(1)) {
            const double wall = std::chrono::duration<double>(now - wall0).count();
            const double sim = sent / actual_rate;
            const double inst = (sent - last_status_sent)
                                / (actual_rate * std::chrono::duration<double>(now - last_status).count());
            size_t qsize;
            {
                std::unique_lock<std::mutex> lk(mtx);
                qsize = queue.size();
            }
            std::fprintf(stderr,
                "[gnss_sim_tx] TX status: sim %.2f s, wall %.1f s, %.3fx realtime, "
                "underflow %llu (short %llu), seq_err %llu, queue %zu\n",
                sim, wall, inst, static_cast<unsigned long long>(async_underflows),
                static_cast<unsigned long long>(short_sends),
                static_cast<unsigned long long>(async_seq_errors), qsize);
            std::fflush(stderr);
            last_status = now;
            last_status_sent = sent;
        }
    }

    // End of burst: one zero guard sample flagged end_of_burst.
    {
        int16_t zero[2] = {0, 0};
        const void* zb[1] = {static_cast<const void*>(zero)};
        size_t items = 0;
        uhd_tx_metadata_handle md = md_eob;
        uhd_tx_streamer_send(streamer, zb, 1, &md, 1.0, &items);
    }

    g_stop.store(true);
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_can_push.notify_all();
    }
    if (reader.joinable()) reader.join();

    // ---- teardown ----------------------------------------------------
    if (have_amd) uhd_async_metadata_free(&amd);
    uhd_tx_metadata_free(&md_start);
    uhd_tx_metadata_free(&md_mid);
    uhd_tx_metadata_free(&md_eob);
    uhd_tx_streamer_free(&streamer);
    uhd_usrp_free(&usrp);

    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();
    const double sim = actual_rate > 0.0 ? sent / actual_rate : 0.0;
    std::fprintf(stderr,
        "[gnss_sim_tx] done: sent %llu samp (%.2f s), wall %.2f s (%.3fx), "
        "underflow %llu (short %llu), seq_err %llu\n",
        static_cast<unsigned long long>(sent), sim, wall,
        wall > 0.0 ? sim / wall : 0.0,
        static_cast<unsigned long long>(async_underflows),
        static_cast<unsigned long long>(short_sends),
        static_cast<unsigned long long>(async_seq_errors));
    std::fflush(stderr);
    return send_error ? 4 : 0;
}
