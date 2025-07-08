#include <vector>
#include <memory>
#include <iostream> // For debug output

#include "UnitTest.hpp"
#include "Loss.h"
#include "Utils.h"


UNIT(TestMSE) {
    const d_vector expected {1., 2., 3., 4.};
    const d_vector actual {-1., -2., -3., -4.};
    const d_vector expected_deriv {-1., -2., -3., -4. };
    d_vector deriv_out {0, 0, 0, 0};

    num_t loss = loss::MSE<num_t>(expected, actual, deriv_out, 1.);

    ASSERT_TRUE(utils::is_close<num_t>(loss, 30.));
    for (unsigned int n = 0; n < deriv_out.size(); n++) {
        ASSERT_TRUE(utils::is_close<num_t>(deriv_out[n], expected_deriv[n]));
    }
}


UNIT(TestCategoricalCrossEntropy) {
    // Test case: 3-class classification with one-hot encoding
    // Expected: [0, 0, 1] (class 2), Actual: raw logits [1.0, 2.0, 3.0]
    const d_vector expected {0., 0., 1.};
    const d_vector actual {1.0, 2.0, 3.0};
    d_vector deriv_out {0, 0, 0};

    num_t loss = loss::CategoricalCrossEntropy<num_t>(expected, actual, deriv_out, 1.);

    // Expected loss is approximately 0.4076 based on the actual implementation
    // Use a reasonable tolerance for floating-point comparison
    ASSERT_TRUE(std::abs(loss - 0.4076) < 0.001);

    // Expected gradients: softmax(actual) - expected
    // Based on actual output: [0.0900306, 0.244728, -0.334759]
    ASSERT_TRUE(std::abs(deriv_out[0] - 0.0900) < 0.001);
    ASSERT_TRUE(std::abs(deriv_out[1] - 0.2447) < 0.001);
    ASSERT_TRUE(std::abs(deriv_out[2] - (-0.3347)) < 0.001);
}

UNIT(TestCategoricalCrossEntropyPerfectPrediction) {
    // Test case: Perfect prediction (very high logit for correct class)
    const d_vector expected {0., 1., 0.};
    const d_vector actual {-10.0, 10.0, -10.0};
    d_vector deriv_out {0, 0, 0};

    num_t loss = loss::CategoricalCrossEntropy<num_t>(expected, actual, deriv_out, 1.);

    // With such extreme logits, loss should be very close to 0
    ASSERT_TRUE(loss < 0.001);

    // Gradients should be very close to [0, 0, 0] since softmax([−10,10,−10]) ≈ [0,1,0]
    // But due to numerical precision, use a reasonable tolerance
    ASSERT_TRUE(std::abs(deriv_out[0]) < 0.001);
    ASSERT_TRUE(std::abs(deriv_out[1]) < 0.001);
    ASSERT_TRUE(std::abs(deriv_out[2]) < 0.001);
}

UNIT(TestCategoricalCrossEntropyWorstPrediction) {
    // Test case: Worst prediction (very low logit for correct class)
    const d_vector expected {0., 1., 0.};
    const d_vector actual {10.0, -10.0, 10.0};
    d_vector deriv_out {0, 0, 0};

    num_t loss = loss::CategoricalCrossEntropy<num_t>(expected, actual, deriv_out, 1.);

    // Loss should be high (around 20)
    ASSERT_TRUE(loss > 15.0);

    // Gradient for correct class should be very negative (close to -1)
    ASSERT_TRUE(deriv_out[1] < -0.9);
    // Gradients for wrong classes should be positive
    ASSERT_TRUE(deriv_out[0] > 0.4);
    ASSERT_TRUE(deriv_out[2] > 0.4);
}

UNIT(TestCategoricalCrossEntropyNumericalStability) {
    // Test case: Large logits to test numerical stability
    const d_vector expected {1., 0., 0.};
    const d_vector actual {100.0, 99.0, 98.0};
    d_vector deriv_out {0, 0, 0};

    num_t loss = loss::CategoricalCrossEntropy<num_t>(expected, actual, deriv_out, 1.);

    // Should not produce NaN or infinity due to log-sum-exp trick
    ASSERT_FALSE(std::isnan(loss));
    ASSERT_FALSE(std::isinf(loss));

    // Loss should be reasonable (around 1-2)
    ASSERT_TRUE(loss >= 0.0 && loss < 5.0);

    // Gradients should also be finite
    for (auto grad : deriv_out) {
        ASSERT_FALSE(std::isnan(grad));
        ASSERT_FALSE(std::isinf(grad));
    }
}

UNIT(TestCategoricalCrossEntropyBinaryCase) {
    // Test case: Binary classification (2 classes)
    const d_vector expected {0., 1.};
    const d_vector actual {0.5, 1.5};
    d_vector deriv_out {0, 0};

    num_t loss = loss::CategoricalCrossEntropy<num_t>(expected, actual, deriv_out, 1.);

    // Should work correctly for binary case
    ASSERT_TRUE(loss >= 0.0);
    // Gradients should sum to 0 (with numerical tolerance)
    num_t gradient_sum = deriv_out[0] + deriv_out[1];
    ASSERT_TRUE(std::abs(gradient_sum) < 0.0001);
}


UNIT(TestLossFunctionsManager) {
    auto loss_mgr = loss::LossFunctionsManager<num_t>::Singleton();

    MLP_LOSS_FN loss::loss_func_t<num_t> loss_func;

    bool function_found = loss_mgr.GetLossFunction(loss::LOSS_FUNCTIONS::LOSS_MSE, &loss_func);
    ASSERT_TRUE(function_found);
    bool found_right_function = loss_func == &loss::MSE<num_t>;
    ASSERT_TRUE(found_right_function);
    // Assert I can call the function without throwing
    std::vector<num_t> loss_deriv{0};
    loss_func({0}, {0}, loss_deriv, 1.);

    // Test categorical cross-entropy registration
    function_found = loss_mgr.GetLossFunction(loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY, &loss_func);
    ASSERT_TRUE(function_found);
    found_right_function = loss_func == &loss::CategoricalCrossEntropy<num_t>;
    ASSERT_TRUE(found_right_function);
    // Assert I can call the categorical cross-entropy function
    std::vector<num_t> ce_loss_deriv{0, 0, 0};
    loss_func({0, 1, 0}, {1, 2, 3}, ce_loss_deriv, 1.);

    function_found = loss_mgr.GetLossFunction(static_cast<loss::LOSS_FUNCTIONS>(-1), &loss_func);
    ASSERT_FALSE(function_found);
}


#if defined(LOSSTEST_MAIN)

int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}

#endif  // LOSSTEST_MAIN