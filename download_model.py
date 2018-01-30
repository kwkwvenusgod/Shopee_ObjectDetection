from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def download_model_from_google_drive(file_id='1o9d3eGp0z_brNnHO9_XNyRNsJMShHRTa', model_path='model_output/model_frcnn.vgg.hdf5'):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    model_file = drive.CreateFile({'id': file_id})
    model_file.GetContentFile(model_path)
    return

