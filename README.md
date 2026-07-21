# Vectorisation raster vers SVG avec DINOv3 et DiffVG

Ce dépôt contient le modèle de vectorisation du projet et une application web qui
l'utilise directement. Le backend actif ne lance plus SuperSVG : il importe la
pipeline réelle de `implementation/`, charge le modèle une seule fois au démarrage,
puis réutilise la même instance pour toutes les requêtes.

## Structure du projet

```text
implementation/
  inference.py                    Pipeline d'inférence complète
  dataset.py                      Lecture RGB, SLIC et préparation des régions
  encoder.py                      Encodeur DINOv3 et tête de chemins
  model.py                        Modèle, rendu DiffVG et export SVG
  checkpoints/                    Checkpoints entraînés
  diffvg/                         Sous-module DiffVG local, avec correctifs Windows

vectorization-app/
  backend/
    main.py                       API FastAPI
    model_service.py              Adaptateur persistant vers implementation/
    config.py                     Configuration .env et chemins pathlib
    SuperSVG/                     Ancien moteur conservé, mais non utilisé
  frontend/                       Interface React/Vite/Tailwind existante
```

## Pipeline utilisée

Le point d'entrée de référence est `implementation/inference.py`. Le backend
réutilise directement ses fonctions `build_model`, `prepare_regions` et
`predict_global_strokes`, ainsi que les fonctions de rendu/export de
`implementation/model.py`.

Pour chaque image :

1. Pillow convertit l'image complète en RGB `float32` dans `[0, 1]`.
2. SLIC segmente l'image ; chaque région est recadrée et redimensionnée en
   `224 x 224`, avec le fond placé à `-1`.
3. Le modèle DINOv3 `vit_small_patch16_dinov3` prédit 128 chemins par région.
4. Les 12 points de chaque chemin fermé (quatre Bézier cubiques) sont remappés
   dans les coordonnées de l'image complète.
5. DiffVG génère le SVG aux dimensions originales et un aperçu PNG sur fond noir.

Le checkpoint utilisé par défaut est :

```text
implementation/checkpoints/raster_to_svg_128paths/epoch_0019.pt
```

Il correspond au modèle `ours_final` évalué dans le dépôt. Ne pas le remplacer
par `raster_to_svg_128paths/latest.pt` : ce dernier contient l'époque 14.

## Prérequis Windows

- Windows 10 ou 11 64 bits ;
- Python 3.11 recommandé ;
- Node.js 18+ (Node.js 20 ou 24 recommandé) et npm ;
- Git avec les sous-modules initialisés ;
- CMake et Visual Studio Build Tools avec le composant C++ pour compiler DiffVG ;
- le checkpoint ci-dessus, ou un autre checkpoint compatible indiqué dans `.env`.

Initialiser DiffVG si le sous-module est vide :

```powershell
git submodule update --init --recursive
```

### GPU et CUDA (RTX 4070 SUPER)

Le GPU est facultatif : `MODEL_DEVICE=auto` utilise CUDA quand
`torch.cuda.is_available()` vaut `True`, sinon le CPU.

La configuration réellement validée sur la machine du projet est : Python
3.11.15, PyTorch 2.5.1, torchvision 0.20.1, runtime PyTorch CUDA 12.4 et une
RTX 4070 SUPER. Cette combinaison est une référence vérifiée, pas une obligation
pour toutes les machines. Pour compiler une extension DiffVG GPU neuve, un CUDA
Toolkit compatible avec le build PyTorch et `nvcc` sont également nécessaires.

Vérification rapide :

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Installation du backend

Depuis PowerShell :

```powershell
cd vectorization-app/backend
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

Installer d'abord la variante PyTorch souhaitée. Pour la configuration CUDA 12.4
validée :

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Pour un backend CPU :

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Installer ensuite le reste des dépendances :

```powershell
python -m pip install -r requirements.txt
```

DiffVG/pydiffvg est une extension native locale et n'a pas de wheel Windows
portable. Après PyTorch, l'installer depuis le sous-module. Pour CUDA :

```powershell
Push-Location ..\..\implementation\diffvg
$env:DIFFVG_CUDA="1"
python setup.py install
Pop-Location
```

Pour une compilation CPU, utiliser `$env:DIFFVG_CUDA="0"`. Vérifier ensuite :

```powershell
python -c "import torch, pydiffvg, diffvg; print(torch.cuda.is_available()); print(pydiffvg.__file__); print(diffvg.__file__)"
```

## Configuration du backend

Créer le fichier local `.env` :

```powershell
Copy-Item .env.example .env
```

Valeurs fournies :

```env
MODEL_CHECKPOINT=../../implementation/checkpoints/raster_to_svg_128paths/epoch_0019.pt
MODEL_DEVICE=auto
IMPLEMENTATION_DIR=../../implementation
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
MAX_UPLOAD_MB=25
```

Les chemins relatifs sont résolus avec `pathlib` depuis
`vectorization-app/backend`. `MODEL_DEVICE` accepte uniquement `auto`, `cpu` ou
`cuda`.

Démarrage avec rechargement du code :

```powershell
python -m uvicorn main:app --reload
```

`python main.py` lit directement `BACKEND_HOST` et `BACKEND_PORT`. Avec la CLI
Uvicorn, ajouter `--host` et `--port` si des valeurs autres que les défauts sont
souhaitées.

## API

- `GET /` : état du serveur ;
- `GET /health` : état du modèle, device et checkpoint ;
- `POST /vectorize` : formulaire multipart contenant `file` (PNG ou JPEG) et
  `num_regions` (entier optionnel entre 2 et 256, valeur par défaut : 64) ;
- `GET /download/{job_id}/{filename}` : téléchargement du SVG ;
- `GET /preview/{job_id}/{filename}` : aperçu PNG produit par la pipeline.

Réponse de vectorisation :

```json
{
  "job_id": "...",
  "svg_filename": "vectorized.svg",
  "download_url": "/download/.../vectorized.svg",
  "preview_filename": "preview.png",
  "preview_url": "/preview/.../preview.png",
  "device": "cuda",
  "num_regions": 64,
  "region_count": 49,
  "path_count": 6272
}
```

Le modèle et le checkpoint sont chargés dans le lifespan FastAPI, une seule fois
par processus. Les inférences utilisent `torch.inference_mode()` et sont
sérialisées par un verrou, car DiffVG conserve son device dans un état global.

## Installation du frontend

```powershell
cd vectorization-app/frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Configuration Vite :

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Sans cette variable, le frontend utilise la même URL par défaut. Le design, le
drag-and-drop, le viewer comparatif, la relance et le téléchargement SVG sont
conservés. Les anciens paramètres SuperSVG ne sont plus envoyés.

Build de production :

```powershell
npm run build
```

## Erreurs courantes

### Checkpoint introuvable

Consulter `http://127.0.0.1:8000/health`. Le champ `checkpoint` montre le chemin
résolu et `error` précise le fichier manquant. Corriger `MODEL_CHECKPOINT` ; depuis
le backend, `implementation/` se trouve à `../../implementation`.

### CUDA indisponible

Utiliser temporairement `MODEL_DEVICE=cpu`, ou vérifier le pilote NVIDIA et la
variante PyTorch installée. `MODEL_DEVICE=cuda` échoue explicitement si
`torch.cuda.is_available()` est faux.

### PyTorch et CUDA incompatibles

Comparer `torch.version.cuda`, la version du pilote affichée par `nvidia-smi` et
la version utilisée pour compiler DiffVG. Réinstaller ensemble PyTorch,
torchvision et l'extension DiffVG dans le même environnement Python.

### pydiffvg ou DiffVG manquant

Une erreur `No module named pydiffvg` ou `No module named diffvg` indique que le
sous-module natif n'est pas installé dans le `venv` actif. Reprendre l'étape
`implementation/diffvg`. Une extension compilée pour une autre version de Python
n'est pas réutilisable.

### CORS

Le backend autorise `localhost` et `127.0.0.1` sur les ports Vite 5173 et 5174.
Utiliser l'une de ces origines, ou ajouter explicitement une autre origine dans
`backend/main.py`.

### Backend non lancé

Si l'interface affiche `Backend is not running`, démarrer Uvicorn, vérifier
`/health`, puis confirmer que `VITE_API_BASE_URL` correspond à l'adresse du
backend.

## Comment démarrer l'application

Backend :

```powershell
cd vectorization-app/backend
.\venv\Scripts\activate
python -m uvicorn main:app --reload
```

Frontend :

```powershell
cd vectorization-app/frontend
npm install
npm run dev
```

Navigateur :

```text
http://localhost:5173
```
