curl -L -o ./lc25000.zip\
  https://www.kaggle.com/api/v1/datasets/download/javaidahmadwani/lc25000 # Descargar el dataset
unzip -o lc25000.zip -d ./ # Descomprimir el dataset
rm -rf lung_colon_image_set/Test\ Set/lung_scc/ # Eliminar la carpeta Test Set/lung_scc
rm -rf lung_colon_image_set/Train\ and\ Validation\ Set/lung_scc/ # Eliminar la carpeta Train and Validation Set/lung_scc
rm lc25000.zip # Eliminar el zip descargado
echo "El dataset ha sido descargado y limpiado correctamente."