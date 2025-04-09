#include "UnitTest.hpp"
#include "Dataset.hpp"
#include <algorithm>

// A simple test case to check the Add function of the Dataset class
UNIT(DatasetAddValidData) {
    Dataset dataset;

    // Prepare some dummy feature and label vectors
    std::vector<float> feature = {1.0f, 2.0f, 3.0f};
    std::vector<float> label = {0.0f};

    // Try adding the feature-label pair to the dataset
    bool result = dataset.Add(feature, label);

    // Check that the Add function returns true (success)
    ASSERT_TRUE(result);

    // Fetch the features and labels from the dataset to verify they were added correctly
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that there is exactly 1 data point in the dataset
    ASSERT_EQ(features->size(), 1);
    ASSERT_EQ(labels->size(), 1);

    // Ensure the feature and label are as expected
    ASSERT_EQ((*features)[0], feature);
    ASSERT_EQ((*labels)[0], label);
}

UNIT(DatasetAddEmptyData) {
    Dataset dataset;

    // Prepare empty feature and label vectors
    std::vector<float> feature = {};
    std::vector<float> label = {};

    // Try adding the empty feature-label pair to the dataset
    bool result = dataset.Add(feature, label);

    // Check that the Add function returns true (success), even for empty data
    ASSERT_TRUE(result);

    // Fetch the features and labels from the dataset to verify they were added correctly
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that there is exactly 1 data point in the dataset, even with empty data
    ASSERT_EQ(features->size(), 1);
    ASSERT_EQ(labels->size(), 1);

    // Ensure the feature and label are empty as expected
    ASSERT_TRUE((*features)[0].empty());
    ASSERT_TRUE((*labels)[0].empty());
}

UNIT(DatasetAddMultipleData) {
    Dataset dataset;

    // Prepare some feature and label vectors
    std::vector<float> feature1 = {1.0f, 2.0f};
    std::vector<float> label1 = {0.0f};
    std::vector<float> feature2 = {4.0f, 5.0f};
    std::vector<float> label2 = {1.0f};

    // Add the first feature-label pair
    dataset.Add(feature1, label1);

    // Add the second feature-label pair
    dataset.Add(feature2, label2);

    // Fetch the features and labels from the dataset to verify they were added correctly
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure there are 2 data points in the dataset
    ASSERT_EQ(features->size(), 2);
    ASSERT_EQ(labels->size(), 2);

    // Ensure the feature and label values match the expected values
    ASSERT_EQ((*features)[0], feature1);
    ASSERT_EQ((*labels)[0], label1);
    ASSERT_EQ((*features)[1], feature2);
    ASSERT_EQ((*labels)[1], label2);
}

// Optionally, you can test for the maximum number of allowed examples if kMax_examples is used
UNIT(DatasetAddMaxExamples) {
    Dataset dataset;

    // Add kMax_examples (100) data points to the dataset
    for (unsigned int i = 0; i < Dataset::kMax_examples; ++i) {
        std::vector<float> feature = {float(i)};
        std::vector<float> label = {float(i % 2)};  // Label alternates between 0.0 and 1.0
        bool result = dataset.Add(feature, label);
        ASSERT_TRUE(result);
    }

    // Fetch the features and labels from the dataset to verify
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that exactly kMax_examples (100) examples have been added
    ASSERT_EQ(features->size(), Dataset::kMax_examples);
    ASSERT_EQ(labels->size(), Dataset::kMax_examples);
}

// A test case to check the Add function with mismatched feature-label lengths
UNIT(DatasetAddFeatureLabelLengthMismatch) {
    Dataset dataset;

    // Prepare a valid feature-label pair
    std::vector<float> valid_feature = {1.0f, 2.0f};
    std::vector<float> valid_label = {0.0f};

    // Add the valid pair to the dataset
    bool result = dataset.Add(valid_feature, valid_label);
    ASSERT_TRUE(result);  // It should succeed for the first pair

    // Prepare an invalid feature-label pair (feature length != label length)
    std::vector<float> invalid_feature = {1.0f};
    std::vector<float> invalid_label = {0.0f, 1.0f};

    // Try adding the invalid pair to the dataset
    result = dataset.Add(invalid_feature, invalid_label);
    ASSERT_FALSE(result);  // It should fail for the second pair due to length mismatch

    // Fetch the features and labels from the dataset to verify only the valid pair was added
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that there is still only 1 valid data point in the dataset
    ASSERT_EQ(features->size(), 1);
    ASSERT_EQ(labels->size(), 1);

    // Ensure the valid feature and label were added
    ASSERT_EQ((*features)[0], valid_feature);
    ASSERT_EQ((*labels)[0], valid_label);
}

// A test case to check when the feature vector length is incorrect
UNIT(DatasetAddFeatureLengthMismatch) {
    Dataset dataset;

    // Prepare a valid feature-label pair
    std::vector<float> valid_feature = {1.0f, 2.0f};
    std::vector<float> valid_label = {0.0f};

    // Add the valid pair to the dataset
    bool result = dataset.Add(valid_feature, valid_label);
    ASSERT_TRUE(result);  // It should succeed for the first pair

    // Prepare an invalid feature vector (different length)
    std::vector<float> invalid_feature = {1.0f};  // Only 1 element instead of 2
    std::vector<float> label = {0.0f};

    // Try adding the invalid feature vector with a valid label vector
    result = dataset.Add(invalid_feature, label);
    ASSERT_FALSE(result);  // It should fail for the second pair due to feature length mismatch

    // Fetch the features and labels from the dataset to verify only the valid pair was added
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that there is still only 1 valid data point in the dataset
    ASSERT_EQ(features->size(), 1);
    ASSERT_EQ(labels->size(), 1);

    // Ensure the valid feature and label were added
    ASSERT_EQ((*features)[0], valid_feature);
    ASSERT_EQ((*labels)[0], valid_label);
}

// A test case to check when the label vector length is incorrect
UNIT(DatasetAddLabelLengthMismatch) {
    Dataset dataset;

    // Prepare a valid feature-label pair
    std::vector<float> feature = {1.0f, 2.0f};
    std::vector<float> valid_label = {0.0f};

    // Add the valid pair to the dataset
    bool result = dataset.Add(feature, valid_label);
    ASSERT_TRUE(result);  // It should succeed for the first pair

    // Prepare an invalid label vector (different length)
    std::vector<float> feature2 = {3.0f, 4.0f};
    std::vector<float> invalid_label = {0.0f, 1.0f};  // Label has 2 elements instead of 1

    // Try adding the feature with the invalid label vector
    result = dataset.Add(feature2, invalid_label);
    ASSERT_FALSE(result);  // It should fail for the second pair due to label length mismatch

    // Fetch the features and labels from the dataset to verify only the valid pair was added
    Dataset::DatasetVector *features;
    Dataset::DatasetVector *labels;
    dataset.Fetch(features, labels);

    // Ensure that there is still only 1 valid data point in the dataset
    ASSERT_EQ(features->size(), 1);
    ASSERT_EQ(labels->size(), 1);

    // Ensure the valid feature and label were added
    ASSERT_EQ((*features)[0], feature);
    ASSERT_EQ((*labels)[0], valid_label);
}


UNIT(DatasetGetFeatures) {
    // Create a small dataset
    Dataset::DatasetVector features = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    Dataset::DatasetVector labels = {
        {0.0f},
        {1.0f}
    };

    // Initialize Dataset object
    Dataset dataset(features, labels);

    // Fetch features with bias
    Dataset::DatasetVector features_with_bias = dataset.GetFeatures(true);

    // Expected output with bias
    Dataset::DatasetVector expected_with_bias = {
        {1.0f, 2.0f, 3.0f, 1.0f},
        {4.0f, 5.0f, 6.0f, 1.0f}
    };

    // Check that the number of feature vectors is correct
    ASSERT_TRUE(features_with_bias.size() == expected_with_bias.size());

    // Check that each feature vector matches the expected output with bias
    for (size_t i = 0; i < expected_with_bias.size(); ++i) {
        ASSERT_TRUE(features_with_bias[i].size() == expected_with_bias[i].size());
        for (size_t j = 0; j < expected_with_bias[i].size(); ++j) {
            ASSERT_TRUE(features_with_bias[i][j] == expected_with_bias[i][j]);
        }
    }

    // Fetch features without bias
    Dataset::DatasetVector features_without_bias = dataset.GetFeatures(false);

    // Expected output without bias
    Dataset::DatasetVector expected_without_bias = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    // Check that the number of feature vectors is correct
    ASSERT_TRUE(features_without_bias.size() == expected_without_bias.size());

    // Check that each feature vector matches the expected output without bias
    for (size_t i = 0; i < expected_without_bias.size(); ++i) {
        ASSERT_TRUE(features_without_bias[i].size() == expected_without_bias[i].size());
        for (size_t j = 0; j < expected_without_bias[i].size(); ++j) {
            ASSERT_TRUE(features_without_bias[i][j] == expected_without_bias[i][j]);
        }
    }
}


// Helper: Sort a DatasetVector (of single-element vectors) by the first element.
static std::vector<std::vector<float>> sortFeatures(const Dataset::DatasetVector &features) {
    auto sorted = features;
    std::sort(sorted.begin(), sorted.end(),
              [](const std::vector<float> &a, const std::vector<float> &b) {
                  return a[0] < b[0];
              });
    return sorted;
}

UNIT(DatasetFIFO) {
    // Enable replay memory, FIFO mode, and set maximum examples to 5.
    Dataset ds;
    ds.ReplayMemory(true);
    ds.SetForgetMode(Dataset::FIFO);
    ds.SetMaxExamples(5);

    // Add 7 examples: feature = {i}, label = {i*10}.
    for (int i = 0; i < 7; ++i) {
        std::vector<float> feature = { static_cast<float>(i) };
        std::vector<float> label   = { static_cast<float>(i * 10) };
        bool added = ds.Add(feature, label);
        ASSERT_TRUE(added);
    }

    // Sample without bias for easier checking.
    auto samplePair = ds.Sample(false);
    auto &sampledFeatures = samplePair.first;
    auto &sampledLabels   = samplePair.second;
    ASSERT_EQ(sampledFeatures.size(), size_t(5));
    ASSERT_EQ(sampledLabels.size(), size_t(5));

    // In FIFO mode, the first two examples (i==0,1) should be removed.
    // The remaining examples (i==2,3,4,5,6) are expected.
    std::vector<float> expected = { 2.f, 3.f, 4.f, 5.f, 6.f };

    // Sort the sampled features so we can compare regardless of sample order.
    auto sortedFeatures = sortFeatures(sampledFeatures);
    ASSERT_EQ(sortedFeatures.size(), expected.size());
    for (size_t i = 0; i < sortedFeatures.size(); ++i) {
        // With bias disabled, each feature vector should have exactly one element.
        ASSERT_EQ(sortedFeatures[i].size(), size_t(1));
        ASSERT_EQ(sortedFeatures[i][0], expected[i]);
    }

    // Check that labels are consistent: label == feature * 10.
    auto sortedLabels = sortFeatures(sampledLabels);
    std::vector<float> expectedLabels;
    for (float f : expected) {
        expectedLabels.push_back(f * 10);
    }
    ASSERT_EQ(sortedLabels.size(), expectedLabels.size());
    for (size_t i = 0; i < sortedLabels.size(); ++i) {
        ASSERT_EQ(sortedLabels[i][0], expectedLabels[i]);
    }
}

UNIT(DatasetRandomEqual) {
    // Test RANDOM_EQUAL mode.
    Dataset ds;
    ds.ReplayMemory(true);
    ds.SetForgetMode(Dataset::RANDOM_EQUAL);
    ds.SetMaxExamples(5);

    for (int i = 0; i < 7; ++i) {
        std::vector<float> feature = { static_cast<float>(i) };
        std::vector<float> label   = { static_cast<float>(i * 10) };
        bool added = ds.Add(feature, label);
        ASSERT_TRUE(added);
    }
    auto samplePair = ds.Sample(false);
    auto &sampledFeatures = samplePair.first;
    auto &sampledLabels   = samplePair.second;
    ASSERT_EQ(sampledFeatures.size(), size_t(5));
    ASSERT_EQ(sampledLabels.size(), size_t(5));

    // For RANDOM_EQUAL, we can't predict which examples remain, but check each pair.
    for (size_t i = 0; i < sampledFeatures.size(); ++i) {
        float f = sampledFeatures[i][0];
        float l = sampledLabels[i][0];
        ASSERT_EQ(l, f * 10);
        ASSERT_TRUE(f >= 0 && f <= 6);
    }
}

UNIT(DatasetRandomOlder) {
    // Test RANDOM_OLDER mode.
    Dataset ds;
    ds.ReplayMemory(true);
    ds.SetForgetMode(Dataset::RANDOM_OLDER);
    ds.SetMaxExamples(5);

    for (int i = 0; i < 7; ++i) {
        std::vector<float> feature = { static_cast<float>(i) };
        std::vector<float> label   = { static_cast<float>(i * 10) };
        bool added = ds.Add(feature, label);
        ASSERT_TRUE(added);
    }
    auto samplePair = ds.Sample(false);
    auto &sampledFeatures = samplePair.first;
    auto &sampledLabels   = samplePair.second;
    ASSERT_EQ(sampledFeatures.size(), size_t(5));
    ASSERT_EQ(sampledLabels.size(), size_t(5));

    for (size_t i = 0; i < sampledFeatures.size(); ++i) {
        float f = sampledFeatures[i][0];
        float l = sampledLabels[i][0];
        ASSERT_EQ(l, f * 10);
        ASSERT_TRUE(f >= 0 && f <= 6);
    }
}

UNIT(DatasetSetMaxExamples) {
    // Test SetMaxExamples() in non-replay-memory mode.
    Dataset ds;
    ds.ReplayMemory(false);
    ds.SetMaxExamples(5);

    // Add exactly 5 examples.
    for (int i = 0; i < 5; ++i) {
        std::vector<float> feature = { static_cast<float>(i) };
        std::vector<float> label   = { static_cast<float>(i * 10) };
        bool added = ds.Add(feature, label);
        ASSERT_TRUE(added);
    }
    // With replay memory disabled, an extra Add() should fail.
    {
        std::vector<float> feature = { 100.f };
        std::vector<float> label   = { 1000.f };
        bool added = ds.Add(feature, label);
        ASSERT_TRUE(!added);
    }
    // Now, reduce the maximum to 3. This should trim the dataset from the end.
    ds.SetMaxExamples(3);
    auto samplePair = ds.Sample(false);
    auto &sampledFeatures = samplePair.first;
    auto &sampledLabels   = samplePair.second;
    ASSERT_EQ(sampledFeatures.size(), size_t(3));
    ASSERT_EQ(sampledLabels.size(), size_t(3));
    // Expect the dataset now contains the first three examples.
    for (size_t i = 0; i < sampledFeatures.size(); ++i) {
        ASSERT_EQ(sampledFeatures[i][0], static_cast<float>(i));
        ASSERT_EQ(sampledLabels[i][0], static_cast<float>(i * 10));
    }
}
