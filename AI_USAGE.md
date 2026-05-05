1. What you used AI for.

For this project, AI was very helpful for outlining steps to successfully complete a latent space diffusion project. In the beginning of the project, I used Chat-GPT to outline the project details and explain to me how I can achieve this. These chats were also helpful for me to understand what was feasible for me and my time/hardware/data constraints. This is how I came to use transfer learning for the VAE and UNet components of this project.

I used copilot GPT-5.2 throughout this project. Using the chat extensively to help me understand what I should build and expain any code it outputs. For example, some high level questions might include, how does DDIM scheduler work, or what kind of noise is injected into the image inputs. Much of the engine was built in tandem with copilot. I guided AI generation throughout the scaffolded engine in order to understand specifically what was being implemented and ensuring it worked cohesively with other parts of the engine. Much of the notebooks were further refined from my scaffolding to produce prettier plots or more organized output.

2. What I did not use AI for.

I did not use AI for much of the notebook content, where I explored the dataset, trained the models, and explored the output/performance. The full_ldm_training notebook is where I spent most of my time, exploring how to use the engine in meaningful ways, load frozen model weights, and run held out images through the network. From the fine tuning on the Div2k dataset, I could tell that the dataset was severly limited and performed quite poorly on portraits. This prompted me to use the FFHQ dataset to fine tune on faces. I also want to better understand how the diffusion steps were effecting output, specifically the point at which quality generation reaches diminishing returns. I also fully guided which plots and images I wanted generated, in order to visualize the model performance and output parameters, such as different noise levels or diffusion steps.

3. How you verified AI output. What did you check? What did you find wrong or suboptimal?
What did you change?

Throughout the engine building I created simple tests for myself in a notebook to ensure that the engine was working properly and produced expected outputs. I also frequently queried the copilot chat to understand what was being generated. AI very often didn't understand that I was running my notebooks through colab resources, making it difficult to understand where files are getting saved. it also would often make edits considering my local system contraints, instead of the colab resources, which often prompted it to use weaker training parameters. Copilot also heavily struggles digesting long notebooks, making it nearly impossible for certain models to produce quality output for later cells. Overall, all code I had generated through copilot, I required some verification output, whether text or another visual, to ensure output is as expected. 

4. What you learned from the interaction.

Diffusion and latent spaces are not something we directly covered in class, so my interactions with AI (and research articles) have been essential for understand this method. I learned the overall design of these diffusion systems, the contraints with this HQ data and complex model, and how to understand diffusion model output/performance.