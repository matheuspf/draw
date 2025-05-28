## TODO


04:30 / 05:30 - run evaluation with different metrics
05:30 / 07:30 - sprout llm work
07:30 / 08:00 - rest #1
08:00 / 10:00 - try reno with different opt params
10:00 / 11:00 - rest #2 (bigger)
11:30 / 13:30 - optimize vqa / aest
13:30 / 14:30 - gym
14:30 / 16:30 - finish sprout work




### Clean up repo

### Improve aesthetic patch score

- Figure out better ways to patch the image
    - Better clip-path
    - Larger regions for much higher score
    - Improve results with lower amount of paths
- Try global transforms that improve aesthetic score


### Improve dataset

- Generate 2k distinct descriptions using multiple LLMs for multiple subjects
    - Half for fitting anything, half only for testing
    - No overlap of subjects or models between training and testing
- Generate questions, options and answers based on the descriptions
    - Use the TIFA repo because it was used for the Kaggle dataset creation too
    - Again generate train and validation with different models
        - Gemmini was used here for the Kaggle dataset
- Generate N images with SD for each question using different prompts
    - Generate more than one image per prompt if that makes sense and is feasible
    - Evaluate and select the best prompts for the SD given the metric after SVG conversion
    - Also select best SVG conversion parameters
- Fit the VQA model with a surrogate smaller model
    - Fit on the question level, so we can run it on much more questions way faster


### Improve ensembling

- Measure performance based on generated vs ground truth questions
- Measure performance based on more questions
    - Add more crops too
- Increase amount of questions using the surrogate VQA model
    - Run on multiple crops as well
- Improve question generation
- Find a better/faster surrogate than just straight up using the metric


## Done

### Improve and accelerate text to image generation

- Research and try other modules

* Use stable diffusion SDXL-Turbo models
* Better than SD 2.1 and extremely fast


### Improve and accelerate image to SVG conversion

- Rewrite the `primitive` lib in C++ with CUDA
- Look for other ways to convert


* Use VTracer with SVG like inputs
* Extremely fast, close but not quite the same as primitive


### Try a different approach

- Research new methods
    - SVGDreamer
    - Chat to SVG

* No other method is quick enough and good enough to deal directly with Text -> SVG
* Most likely, almost certainly, other teams are using SD with some other method to convert to SVG


### Ensemble

- Run multiple image generation + SVG conversions and test against the metric

* Results do get better, but very marginally even with full blown evaluation metric (10b-448 VQA model)
* Options are generating better and more questions, running with augmentations - significant difference for aesthetic score

