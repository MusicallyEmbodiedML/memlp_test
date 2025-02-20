#include <iostream>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"

#include "microunit.h"
#include "easylogging++.h"


// Include unit tests as CPP!
#include "test/MLPTest.cpp"
#include "test/LayerTest.cpp"
#include "test/NodeTest.cpp"
#include "test/LossTest.cpp"
#include "test/SerialiseTest.cpp"
#include "test/DatasetTest.cpp"


int main()
{
    stdio_init_all();
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
