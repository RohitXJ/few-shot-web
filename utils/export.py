import torch
import os
import zipfile
def export_model(run_config, export_dir="export/", filename="fewshot_model.pt", zip_output=False):
    os.makedirs(export_dir,exist_ok=True)
    model_path = os.path.join(export_dir,filename)
    if not zip_output: 
        torch.save(run_config,model_path)
        print("Model Exported")
        return model_path
    else:
        zip_path = model_path.replace(".pt",".zip")
        with zipfile.ZipFile(zip_path,'w') as zipf:
            zipf.write(model_path, arcname=filename)
        os.remove(model_path)
        print("Model Exported")
        return zip_path
