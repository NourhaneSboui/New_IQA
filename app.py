import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tifffile
import torch
from torchvision.transforms.functional import to_tensor
from IQADataset import NonOverlappingCropPatches

# Import your model and IQAPerformance class here
from Network import CNNIQAnet
from training import IQAPerformance

class IQAApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.title("Image Quality Assessment")

        # Create a label to display the image
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Create a button to load an image
        self.load_button = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Create a button to perform image quality assessment
        self.assess_button = tk.Button(self, text="Assess Image Quality", command=self.assess_quality)
        self.assess_button.pack()
        # Create a button to perform image quality assessment
        self.assess_button = tk.Button(self, text="Application's classification", command=self.assess_quality)
        self.assess_button.pack()
        # Load the trained model
        model = CNNIQAnet()
        model.load_state_dict(torch.load('C:/Users/win 10/Desktop/CNNIQA/CNNIQA/results/CNNIQA-EuroSat-EXP0-lr=0.001'))
        model.eval()

        # Initialize the IQAPerformance class
        self.performance_metrics = IQAPerformance()

    def load_image(self):
        # Open a file dialog to choose an image
        file_path = filedialog.askopenfilename(filetypes=[("TIFF Files", "*.tif;*.tiff")])
        if file_path:
            
            # Load the image using tifffile
            image = tifffile.imread(file_path)

            # Display the image in the GUI
            image = Image.fromarray(image)
            image.thumbnail((400, 400))  # Resize the image to fit the label
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Convert the image to tensor for assessment
            self.input_image = image.convert("L")  # Add batch dimension

    def assess_quality(self):
        model = CNNIQAnet()
        model.load_state_dict(torch.load('C:/Users/win 10/Desktop/CNNIQA/CNNIQA/results/CNNIQA-EuroSat-EXP0-lr=0.001'))
        model.eval()
        if hasattr(self, 'input_image'):
            # Perform image quality assessment using the model
            with torch.no_grad():
                im = self.input_image
                patches = NonOverlappingCropPatches(im, 32, 32)
                patch_scores = model(torch.stack(patches).to(torch.device('cpu')))
                '''output = self.model(self.input_image)'''
                predicted_score = patch_scores.mean().item()

            # Assuming you have the ground truth score for the input image
            ground_truth_score = 8.8371  # Replace with the actual ground truth score

            # Calculate the difference between predicted and ground truth scores
            difference = abs(predicted_score - ground_truth_score)

            # Display the result in a message box
            tk.messagebox.showinfo("Image Quality Assessment",
                                   f"Predicted Score: {predicted_score}\n"
                                   f"Ground Truth Score: {ground_truth_score}\n"
                                   f"Difference: {difference}")

    def run(self):
        self.mainloop()

if __name__ == "__main__":
    app = IQAApp()
    app.run()
