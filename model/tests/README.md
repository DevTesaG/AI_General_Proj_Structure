# Test example cases

Different deep learning scenarios and parts of the codebase when unit testing can be incredibly useful. Ensuring that our data has the right format is critical. A few others are: 

## Data
- Ensure that our data has the right format (yes I put it again here for completion)
- Ensure that the training labels are correct
- Test our complex processing steps such as image manipulation
- Assert data completion, quality, and errors
- Test the distribution of the features

## Training
- Run a training step and compare the weight before and after to ensure that they are updated
- Check that our loss function can be actually used on our data

## Evaluation
- Having tests to ensure that your metrics ( e.g accuracy, precision, and recall ) are above a threshold when iterating over different architectures
- You can run speed/benchmark tests on training to catch possible overfitting
- Of course, cross-validation can be in the form of a unit test

## Model Architecture
- The model’s layers are actually stacking
- The model’s output has the correct shape

# Integration/Acceptance tests
Something that I deliberately avoided mentioning is integration and acceptance tests. These kinds of tests are very powerful tools and aim to test how well our system integrates with other systems. If you have an application with many services or client/server interaction, acceptance tests are the go-to functionality to make sure that everything works as expected at a higher level.

Later throughout the course, when we deploy our model in a server, we will absolutely need to write some acceptance tests as we want to be certain that the model returns what the user/client expects in the form that he expects it. As we iterate over our application while it is live and is served to users, we can’t have a failure due to some silly bug (remember the reliability principle from the first article?) These kinds of things acceptance tests help us avoid.