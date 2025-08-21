#include <iostream>

#ifndef LINUX
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#endif

#include "microunit.h"
#include "easylogging++.h"


// Include unit tests as CPP!
#include "test/MLPTest.cpp"
#include "test/LayerTest.cpp"
#include "test/NodeTest.cpp"
#include "test/LossTest.cpp"
#include "test/SerialiseTest.cpp"
#include "test/DatasetTest.cpp"

#ifdef LINUX

int main(int [[maybe_unused]] argc, char* [[maybe_unused]] argv[])
{
    // Initialise the logger
    START_EASYLOGGINGPP(argc, argv);

    // Run unit tests
    bool tests_passed = microunit::UnitTester::Run();

    return tests_passed ? 0 : 1;  // Return 0 if all tests passed, otherwise 1
}

#else

int main()
{
    stdio_init_all();

    // For Pico, create dummy argc/argv for easylogging
    int argc = 1;
    char* argv[] = {(char*)"memlp_test", nullptr};
    START_EASYLOGGINGPP(argc, argv);

    // Initialise the Wi-Fi chip
    if (cyw43_arch_init()) {
        printf("Wi-Fi init failed\n");
        return -1;
    }

    // Waiting time...
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
    for (unsigned int n = 3; n > 0; n--) {
        std::cout << "Waiting " << n <<
            " second" << ((n>1 ? "s" : ""))<<
            " before unit test start..." << std::endl;
        sleep_ms(1000);
    }
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);

    // Run unit tests
    bool tests_passed = microunit::UnitTester::Run();

    while (true) {
        if (tests_passed) {
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
            sleep_ms(500);
        }
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(500);
    }

    return 0;  // Should never return
}

#endif // LINUX
