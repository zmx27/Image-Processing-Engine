# The network layer (docs/PLAN.md Phase 4, Track N): wire codec, sockets, the threaded
# server and the reference client. Portable — it proves protocol, concurrency and
# backpressure against the CPU backend with zero CUDA code, so a bug found here is never
# confused with a CUDA bug.
#
# It links imgjit_io for one reason only: the flags-bit-1 server-side write path emits a
# PNG (docs/PROTOCOL.md). Nothing here ever DECODES an image — network payloads are raw
# contiguous pixels and stb never sees them.
add_library(imgjit_net STATIC
  client.cpp
  protocol.cpp
  server.cpp
  socket.cpp)
target_link_libraries(imgjit_net PUBLIC imgjit_core imgjit_util PRIVATE imgjit_io)
target_link_libraries(imgjit_net PUBLIC Threads::Threads)
