# SmartFew

SmartFew is your swiss knife for semi-supervised structuring of unlabeled data. 

How it works:

1. Prepare the file with image URLs _image_urls.txt_, e.g.
    ```text
    https://myhost.com/image1.jpg
    https://myhost.com/image2.jpg
    ...
    ```
2. Run server
    ```bash
    cd server && python start.py --input image_urls.txt
    ```

3. Go to `http://localhost:14321/` in your browser and start selecting images. Then press **Submit** to continue with a new trial.
The underlying process starts to learn your selection, and you are expecting to see more and more relevant results in your consequent trials.

The algorithm is powered by [Few Shot learning](https://msiam.github.io/Few-Shot-Learning/), that gives an opportunity to learn very fast and quickly adapts to unseen tasks. 

