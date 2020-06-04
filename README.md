# Knowledge Extraction Noise

_An attempt to transfer knowledge from one model to another with noise._

**What is this?**  
Inspired by knowledge distillation, which involves teaching a model (student model) to act like another model (teacher model). This is done by mimicking the teacher's output on input data. I wondered if the same could be achieved with just random noise.

**What is it not?**  
The point is not to merely copy the weights, it is just about achieving the same decision boundaries.

**Does it work?**  
[It depends](https://paulvinell.github.io/research/2020/06/01/knowledge-extraction-noise.html); it often decreases validation loss a little (before increasing it), sometimes it can increase validation accuracy. Despite the fact that it doesn't work particularly well, the code might be fun to look at anyway.

**How does it work?**  
There are two training variants in the code.
1. Random noise.
2. Adversarial random noise. This noise fools the teacher into believing that the noise depicts some object category with a high probability.

**Model**  
The model used is ResNet-12 taken from [here](https://keras.io/examples/cifar10_resnet/).
