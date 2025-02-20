#include "UnitTest.hpp"
#include "Dataset.hpp"

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
