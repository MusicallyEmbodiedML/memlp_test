#include "UnitTest.hpp"
#include "utils/Serialise.hpp"
#include "easylogging++.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>

void dump_buffers(const unsigned char *expected,
        const unsigned char *actual, size_t size) {

    std::printf("~~~ Buffer dump ~~~\n");

    size_t rows = size >> 2;
    size_t last_row = size - (rows << 2);
    // All rows but last
    for (unsigned int row = 0; row < rows; row++) {
        size_t elements_consumed = (row << 2);
        const unsigned char *exp_ptr = expected + elements_consumed;
        const unsigned char *act_ptr = actual + elements_consumed;
        std::printf("- Positions %d-%d -\n", elements_consumed, elements_consumed+4);
        std::printf("Exp: 0x%02x 0x%02x 0x%02x 0x%02x\n",
            *exp_ptr, *(exp_ptr+1), *(exp_ptr+2), *(exp_ptr+3));
        std::printf("Act: 0x%02x 0x%02x 0x%02x 0x%02x\n",
            *act_ptr, *(act_ptr+1), *(act_ptr+2), *(act_ptr+3));
    }
    // Last row
    size_t elements_consumed = (rows<<2);
    if (elements_consumed < size) {
    std::printf("- Row %d -\n", elements_consumed+2);
    const unsigned char *exp_ptr = expected + elements_consumed;
    const unsigned char *act_ptr = actual + elements_consumed;
    std::printf("Exp: ");
    for (unsigned int n = 0; n < last_row; n++) {
        std::printf("0x%02x", *(exp_ptr++));
    }
    std::printf("\n");
    std::printf("Act: ");
    for (unsigned int n = 0; n < last_row; n++) {
        std::printf("0x%02x", *(act_ptr++));
    }
    std::printf("\n");
    }

    std::printf("~~~~~~~~~~~~~~~~~~~\n");
}

UNIT(SerialiseFromVector2d) {

    // Serialise this vector
    const std::vector< std::vector<float> > vec_in {
        {1, 2, 3,},
        {4, 5, 6,},
    };
    std::vector<uint8_t> buffer;
    const std::vector<uint8_t> expected = {
        0x02, 0x00, 0x00, 0x00,  // col = 2
        0x03, 0x00, 0x00, 0x00,  // row = 3
        0x00, 0x00, 0x80, 0x3f,  // 1
        0x00, 0x00, 0x00, 0x40,  // 2
        0x00, 0x00, 0x40, 0x40,  // 3
        0x00, 0x00, 0x80, 0x40,  // 4
        0x00, 0x00, 0xa0, 0x40,  // 5
        0x00, 0x00, 0xc0, 0x40   // 6
    };

    size_t w_head = 0;
    w_head = Serialise::FromVector2D(w_head, vec_in, buffer);

    ASSERT_TRUE(w_head == buffer.size());
    ASSERT_TRUE(expected.size() == buffer.size());
    auto buffer_iterator = buffer.begin();
    unsigned int n = 0;
    for (auto &expd : expected) {
        bool comparison = (*buffer_iterator == expd);
        if (!comparison) {
            std::printf("Buffers not equal at position %d, expected 0x%02x, actual 0x%02x.\n",
                n, expd, *buffer_iterator);
        }
        ASSERT_TRUE(comparison);
        buffer_iterator++;
        n++;
    }

    // Then this vector after that
    const std::vector< std::vector<float> > vec_in_2 {
        {7, 8,},
        {9,10,},
    };
    const std::vector<uint8_t> expected_2 = {
        0x02, 0x00, 0x00, 0x00,  // col = 2
        0x03, 0x00, 0x00, 0x00,  // row = 3
        0x00, 0x00, 0x80, 0x3f,  // 1
        0x00, 0x00, 0x00, 0x40,  // 2
        0x00, 0x00, 0x40, 0x40,  // 3
        0x00, 0x00, 0x80, 0x40,  // 4
        0x00, 0x00, 0xa0, 0x40,  // 5
        0x00, 0x00, 0xc0, 0x40,  // 6
        0x02, 0x00, 0x00, 0x00,  // col = 2
        0x02, 0x00, 0x00, 0x00,  // row = 2
        0x00, 0x00, 0xe0, 0x40,  // 7
        0x00, 0x00, 0x00, 0x41,  // 8
        0x00, 0x00, 0x10, 0x41,  // 9
        0x00, 0x00, 0x20, 0x41,  // 10
    };

    w_head = Serialise::FromVector2D(w_head, vec_in_2, buffer);

    ASSERT_TRUE(w_head == buffer.size());
    ASSERT_TRUE(expected_2.size() == buffer.size());

    dump_buffers(expected_2.data(), buffer.data(), expected_2.size());

    buffer_iterator = buffer.begin();
    n = 0;
    for (auto &expd : expected_2) {
        bool comparison = (*buffer_iterator == expd);
        if (!comparison) {
            std::printf("Buffers not equal at position %d, expected 0x%02x, actual 0x%02x.\n",
                n, expd, *buffer_iterator);
        }
        ASSERT_TRUE(comparison);
        buffer_iterator++;
        n++;
    }
}

UNIT(SerialiseToVector2d) {

    #if !(__XS3A__)
        std::printf("Test not working on x86.\n");
        ASSERT_TRUE(0);
        return;
    #endif

    const std::vector<uint8_t> buffer = {
        0x02, 0x00, 0x00, 0x00,  // col = 2
        0x03, 0x00, 0x00, 0x00,  // row = 3
        0x00, 0x00, 0x80, 0x3f,  // 1
        0x00, 0x00, 0x00, 0x40,  // 2
        0x00, 0x00, 0x40, 0x40,  // 3
        0x00, 0x00, 0x80, 0x40,  // 4
        0x00, 0x00, 0xa0, 0x40,  // 5
        0x00, 0x00, 0xc0, 0x40   // 6
    };
    const std::vector< std::vector<float> > expected {
        {1, 2, 3,},
        {4, 5, 6,},
    };
    std::vector< std::vector<float> > vec_out;

    size_t r_head = 0;
    r_head = Serialise::ToVector2D<float>(r_head, buffer, vec_out);

    ASSERT_TRUE(r_head == buffer.size());
    ASSERT_TRUE(expected.size() == vec_out.size());
    ASSERT_TRUE(expected[0].size() == vec_out[0].size());
    ASSERT_TRUE(expected == vec_out);
}

#if 0
int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}
#endif
