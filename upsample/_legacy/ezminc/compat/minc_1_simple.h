/*
 * minc_1_simple.h -- stand-in for MINC's simple reader/writer, so the EZminc
 * tools compile unmodified.
 *
 * The four vendored tools (mincnlm, minc_anlm, volume_dwt) and the shared
 * noise_estimate.cpp include this for their file I/O, but every call to the
 * reader/writer lives in a command-line main() that the build renames away:
 * the shim drives the underlying routines in-process, on caller-owned
 * buffers, and never touches a MINC file.  These stubs therefore compile but
 * abort if anything ever reaches them.
 *
 * Two things must match the real MINC API closely enough for the dead mains
 * to compile: the type names (nc_type, VIO_Status, VIO_BOOL,
 * minc_input_options) and the member signatures of minc_1_reader,
 * minc_1_writer and minc_info.  VIO_Real / VIO_OK / TRUE / FALSE / VIO_ABS
 * are left to the volume_io.h stand-in that nl_means.h pulls in.
 */

#ifndef N3_EZMINC_COMPAT_MINC_1_SIMPLE_H
#define N3_EZMINC_COMPAT_MINC_1_SIMPLE_H

#include <stdio.h>
#include <stdlib.h>

/* netcdf's nc_type and NC_* tags, as MINC headers assume them. */
typedef int nc_type;
#define NC_BYTE 1
#define NC_CHAR 2
#define NC_SHORT 3
#define NC_INT 4
#define NC_LONG NC_INT
#define NC_FLOAT 5
#define NC_DOUBLE 6

/* volume_io abbreviations mincnlm.cpp's dead main() declares.  The
 * volume_io.h stand-in defines VIO_OK / VIO_Real; VIO_Status and VIO_ERROR
 * are only mentioned here. */
typedef int VIO_Status;
#ifndef VIO_ERROR
#define VIO_ERROR (-1)
#endif
typedef int VIO_BOOL;

typedef struct {
  char _unused;
} minc_input_options;

static void n3_ezminc_no_file_io(const char *what)
{
  fprintf(stderr, "torch_n3 legacy shim: %s is not available\n", what);
  abort();
}

namespace minc {

  struct minc_diminfo {
    double start;
    double step;
    int length;
  };

  class minc_info {
  public:
    minc_diminfo &operator[](int i) { return _dims[i]; }
    const minc_diminfo &operator[](int i) const { return _dims[i]; }
  private:
    minc_diminfo _dims[5];
  };

  class minc_1_reader {
  public:
    void open(const char *) { n3_ezminc_no_file_io("minc_1_reader::open"); }
    int dim_no() const { n3_ezminc_no_file_io("minc_1_reader::dim_no"); }
    int ndim(int) const { n3_ezminc_no_file_io("minc_1_reader::ndim"); }
    double nspacing(int) const {
      n3_ezminc_no_file_io("minc_1_reader::nspacing");
    }
    minc_info info() const {
      n3_ezminc_no_file_io("minc_1_reader::info");
    }
    nc_type datatype() const {
      n3_ezminc_no_file_io("minc_1_reader::datatype");
    }
    int map_space(int) const { n3_ezminc_no_file_io("minc_1_reader::map_space"); }
    void setup_read_float() { n3_ezminc_no_file_io("minc_1_reader::setup_read_float"); }
  };

  class minc_1_writer {
  public:
    void open(const char *, const minc_info &, int, nc_type) {
      n3_ezminc_no_file_io("minc_1_writer::open");
    }
    void copy_headers(minc_1_reader &) {
      n3_ezminc_no_file_io("minc_1_writer::copy_headers");
    }
    void append_history(char *) { n3_ezminc_no_file_io("minc_1_writer::append_history"); }
    void setup_write_float() {
      n3_ezminc_no_file_io("minc_1_writer::setup_write_float");
    }
  };

}

#endif /* N3_EZMINC_COMPAT_MINC_1_SIMPLE_H */