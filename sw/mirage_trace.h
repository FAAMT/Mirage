/* =====================================================================
 * mirage_trace.h -- open/close for the tracing backend (io_trace.c)
 *
 * Host builds only. Nothing in the driver includes this.
 * ===================================================================== */
#ifndef MIRAGE_TRACE_H
#define MIRAGE_TRACE_H

int  mirage_trace_open(const char *path);
void mirage_trace_close(void);   /* flushes and writes END */

#endif /* MIRAGE_TRACE_H */
