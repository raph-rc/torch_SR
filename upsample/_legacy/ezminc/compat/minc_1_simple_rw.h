/*
 * minc_1_simple_rw.h -- stand-in for MINC's simple_volume save/load, so the
 * EZminc tools compile unmodified.
 *
 * Like minc_1_simple.h, the only callers are the renamed-away mains; the shim
 * feeds the same simple_volume<T> buffers in and out in-process.  The four
 * templates below are instantiated (and therefore must compile) but never
 * called, so they abort if they ever are.
 */

#ifndef N3_EZMINC_COMPAT_MINC_1_SIMPLE_RW_H
#define N3_EZMINC_COMPAT_MINC_1_SIMPLE_RW_H

#include <minc_io_simple_volume.h>
#include <minc_1_simple.h>

namespace minc {

  template<class T>
  void load_simple_volume(minc_1_reader &rdr, simple_volume<T> &vol)
  {
    (void) rdr; (void) vol;
    n3_ezminc_no_file_io("load_simple_volume");
  }

  template<class T>
  void load_standard_volume(minc_1_reader &rdr, T *buffer)
  {
    (void) rdr; (void) buffer;
    n3_ezminc_no_file_io("load_standard_volume");
  }

  template<class T>
  void save_simple_volume(minc_1_writer &wrt, const simple_volume<T> &vol)
  {
    (void) wrt; (void) vol;
    n3_ezminc_no_file_io("save_simple_volume");
  }

  template<class T>
  void save_standard_volume(minc_1_writer &wrt, const T *buffer)
  {
    (void) wrt; (void) buffer;
    n3_ezminc_no_file_io("save_standard_volume");
  }

}

#endif /* N3_EZMINC_COMPAT_MINC_1_SIMPLE_RW_H */