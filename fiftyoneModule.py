import kagglehub
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Download latest version
path = kagglehub.dataset_download("robinreni/revitsone-5class")
'''
dataset = fo.Dataset.from_dir(
    dataset_dir=path+"/revitsone-5classes/Revitsone-5classes",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name="revitsone-5class",
    overwrite=True
)
'''
dataset = fo.load_dataset("revitsone-5class")

print(dataset)

#session = fo.launch_app(dataset)

model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
embeddings = dataset.compute_embeddings(model)

results = fob.compute_visualization(
    dataset, embeddings=embeddings, seed=51, brain_key="img_viz"
)

session = fo.launch_app(dataset)
