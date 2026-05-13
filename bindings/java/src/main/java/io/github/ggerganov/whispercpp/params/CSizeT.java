package io.github.ggerganov.whispercpp.params;

import com.sun.jna.IntegerType;
import com.sun.jna.Native;

// Platform-correct mapping for C `size_t`. JNA's NativeLong is wrong on
// Windows x64, where `long` is 32-bit but `size_t` is 64-bit (LLP64).
public class CSizeT extends IntegerType {
    public static final int SIZE = Native.SIZE_T_SIZE;

    public CSizeT() {
        this(0);
    }

    public CSizeT(long value) {
        super(SIZE, value, true);
    }
}
