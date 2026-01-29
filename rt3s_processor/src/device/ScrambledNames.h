/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef GPU_AUDIO_RT3S_SCRAMBLED_NAMES_INCLUDED
#define GPU_AUDIO_RT3S_SCRAMBLED_NAMES_INCLUDED

#pragma once

// The entries in GPUFUNCTIONS_SCRAMBLED are used to replace the processor device function names
// during compilation to avoid name conflicts between processors.
// New processors must always define their own unique names in this list.
// clang-format off
#define GPUFUNCTIONS_SCRAMBLED \
IyMpYK7Q2nMPy00L7DXL, \
WWILqIUuFfZbAOyOZQ3B, \
k4mmJwr6cJFQvKSrVEq3, \
W23MmQwyN9NAqBVSLovy, \
Chu0TiY4RsHPihoBMGI9, \
rcdyM615exnVpDvkKaqE, \
j2x7s1GT8o7So7X283Yi, \
pdYT757GpinShQwRRoec, \
D8vLjuAaKBlVFf6Ty793, \
jDUxbNKigO1BMG3IkTjp, \
BOPAlxHmLyknKFXigo4X, \
mDzJLbuPXnk5GRussF9Q, \
hvqnSiD8ZwsaHlISzw6Q, \
K15kEZ5TUjzLNwlCfjdQ, \
ZeOc9om06XdrmwDuJbZO, \
srNVbErnJv88gg1ogOw9, \
gdCwe35iLI6CuZE4DBOX, \
BpNrgDtPSpCAvsdZ4NUQ, \
MIehfUXVg5CTtJOBqwNj, \
vthsR4lO8GJtnlyeoAXM, \
ys7TXjjpxSnN6Bo56vnt, \
nGG0TwdOOYPlCSV1QvIb, \
zuSqPCCafuhgWOCxAJES, \
dpmWY6LHow5cQhJg5ErX, \
xq0Zs14JDH5lFUPv59VR, \
IvEaY3wpzl806YiHt7Hs, \
N5leVWWz8dWd2sCEW6Bj, \
u3GUagDC6PEt5x2F6Nn5, \
W07FMTV1fTsSIMtmmYU7, \
vVDcOKPSvvNATcJUNFmx, \
nSzmXL65fKQ5pDnDhwRX, \
W7DTONR2RoBz3JfBFCrh, \
sbieMFU9gg5HSfJZJ90X, \
dsEUuGHXCxXzCt0Or1dT, \
gBau3wIUb6hncZ493qRr, \
LXO0ToiQDn0TkAgV21ec, \
QtrAgu7u3X6YTTUAKuiH, \
WJbTAWWEhYRIp8jlKsS8, \
nbvTUxNmTZHeiEWELWbC, \
SeogVy8UTPxtjjFc7Gyj, \
V7nIMG8Wx3qiy8ePuY2T, \
U8KKWIcktJhXMjiOSe2H, \
QBVElYV2PdR6uvZdgiit, \
z6x84kotovJDo5dP3rfb, \
to3TqsOO2ZwolrLxX9tD, \
nqQNppmHkGsT5SKdF7SE, \
oWoMGL0v6JnKVcRzP94W, \
TxKYpue0DUFhBD2zYLmX, \
d8rZTWcubizvxL06Cs54, \
ThzRHnoF1KmEqVG6xpwY, \
LI4XFxaSS4yIH6xMFIBu, \
VMGYiARJXlFS3m5LZYwe, \
R9iKN9KNRYwpLYWkRf4Q, \
b2evIDAciEo3zqiYWLwF, \
cADPaq2rcgBRrPbO7kOB
// clang-format on

#if !defined(GPU_AUDIO_MAC) && !defined(GPU_AUDIO_RTC_ONLINE)
#include <cstdint>
#endif

// DO NOT REMOVE! Contains macros for device function name substitution.
#include <scheduler/common_macros.h>

#endif /* GPU_AUDIO_RT3S_SCRAMBLED_NAMES_INCLUDED */
