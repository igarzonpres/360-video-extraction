conda env create -f environment.yml

conda activate 360pipeline

python run_gui.py


New: Time Range Selection
- In the GUI, two fields appear above the START SPLITTING / START ALIGNING buttons: Start (hh:mm:ss) and End (hh:mm:ss).
- Leave Start empty to begin at 00:00:00. Leave End empty to process until the end of each video.
- If End is not strictly greater than Start, the app warns you and does not start.
- Only the specified range is extracted during the splitting stage; subsequent stages operate on those frames.



Convenciones: yaw=0° en el centro horizontal del equirectangular; yaw crece hacia la derecha; 180° es atrás. El mapeo usa $u=(1+\text{yaw}/\pi)/2$. Pitch positivo inclina hacia abajo; negativo hacia arriba. El mapeo usa $\text{pitch}=-\arctan2(r_y,\|r_{xz}\|)$ y $v=(1-\text{pitch}\cdot2/\pi)/2$.&#x20;

* Pano_camera0: (0, 90) → horizontal; derecha 90°.&#x20;
* Pano_camera1: (32, 0) → abajo 32°; frente.&#x20;
* Pano_camera2: (-42, 0) → arriba 42°; frente.&#x20;
* Pano_camera3 (0, 42) → horizontal; derecha 42°.&#x20;
* Pano_camera4: (0, -25) → horizontal; izquierda 25°.&#x20;
* Pano_camera5: (42, 180) → abajo 42°; atrás.&#x20;
* Pano_camera6: (-32, 180) → arriba 32°; atrás.&#x20;
* Pano_camera7: (0, 205) → horizontal; atrás-izquierda (entre 180° y 270°).&#x20;
* Pano_camera8: (0, 138) → horizontal; atrás-derecha (entre 90° y 180°).&#x20;

Con el pitch se controla el "cabeceo" de la imagen. Con pitch 0 las imágenes están horizontales, es decir, mirando al horizonte. 

    * Pitch positivo → inclina hacia abajo.
    * Pitch negativo → inclina hacia arriba (techo).

Acercando el pitch al 0 conseguimos que las cámaras virtuales renderizadas queden más planas, más mirando hacia el horizonte, mientras que alejándolas del 0 hacia |90| las inclinamos hacia arriba o hacia abajo (dependiendo del signo)
