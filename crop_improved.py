import os
import json
from PIL import Image
import numpy as np

# Chemins
IMG_DIR = "images/fashionpedia/train"
ANN_FILE = "images/fashionpedia/instances_attributes_train2020.json"
CROP_DIR = "images/fashionpedia/crops_improved"

with open(ANN_FILE, "r") as f:
    anns = json.load(f)

articles_db = []

def improve_bbox(bbox, img_width, img_height, padding_ratio=0.05, min_size=64):
    """
    Améliore une bounding box en ajoutant du padding et en vérifiant la taille minimale
    
    Args:
        bbox: [x, y, w, h]
        img_width, img_height: dimensions de l'image originale
        padding_ratio: ratio de padding à ajouter (0.15 = 15% de chaque côté)
        min_size: taille minimale en pixels
    """
    x, y, w, h = bbox
    
    # Filtrer les bboxes trop petites
    if w < min_size or h < min_size:
        return None
    
    # Calculer le padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    
    # Nouvelles coordonnées avec padding
    new_x = max(0, x - pad_x)
    new_y = max(0, y - pad_y)
    new_x2 = min(img_width, x + w + pad_x)
    new_y2 = min(img_height, y + h + pad_y)
    
    # Nouvelles dimensions
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y
    
    # Vérifier que c'est encore assez grand
    if new_w < min_size or new_h < min_size:
        return None
    
    return [new_x, new_y, new_w, new_h]

def crop_with_square_aspect(image, bbox, target_size=224):
    """
    Crop l'image en gardant un aspect ratio carré et en redimensionnant
    """
    x, y, w, h = bbox
    
    # Prendre le max entre w et h pour faire un carré
    size = max(w, h)
    
    # Centrer le carré
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Nouvelles coordonnées pour le carré
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)
    new_x2 = min(image.width, new_x + size)
    new_y2 = min(image.height, new_y + size)
    
    # Ajuster si on dépasse les bords
    if new_x2 - new_x < size:
        new_x = max(0, new_x2 - size)
    if new_y2 - new_y < size:
        new_y = max(0, new_y2 - size)
    
    crop = image.crop((new_x, new_y, new_x2, new_y2))
    
    # Redimensionner à la taille cible
    if crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return crop

# Créer le dossier de destination
os.makedirs(CROP_DIR, exist_ok=True)

print("Amélioration du cropping...")
processed = 0
skipped = 0

# Traiter chaque image
for img_info in anns["images"][:500]:  # Limiter à 100 images pour le test
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = os.path.join(IMG_DIR, file_name)
    
    if not os.path.exists(img_path):
        continue
    
    try:
        with Image.open(img_path) as im:
            img_width, img_height = im.size
            
            # Objets annotés dans cette image
            objs = [o for o in anns["annotations"] if o["image_id"] == img_id]
            
            for obj in objs:
                cat_id = obj["category_id"]
                attrs = obj.get("attributes", [])
                bbox = obj["bbox"]  # [x, y, w, h]
                
                # Améliorer la bbox
                improved_bbox = improve_bbox(bbox, img_width, img_height, 
                                           padding_ratio=0.02, min_size=60)
                
                if improved_bbox is None:
                    skipped += 1
                    continue
                
                # Crop avec aspect carré
                crop = crop_with_square_aspect(im, improved_bbox, target_size=224)
                
                # Vérifier que le crop est valide
                if crop.width == 0 or crop.height == 0:
                    skipped += 1
                    continue
                
                # Sauvegarder
                crop_path = os.path.join(CROP_DIR, f"{img_id}_{obj['id']}.jpg")
                crop.save(crop_path, "JPEG", quality=90)
                
                articles_db.append({
                    "id": f"{img_id}_{obj['id']}",
                    "cat": cat_id,
                    "attrs": attrs,
                    "img": crop_path,
                    "original_bbox": bbox,
                    "improved_bbox": improved_bbox
                })
                
                processed += 1
                
                if processed % 100 == 0:
                    print(f"Traité: {processed}, Ignoré: {skipped}")
    
    except Exception as e:
        print(f"Erreur avec {img_path}: {e}")
        continue

print(f"Terminé! {processed} crops améliorés créés, {skipped} ignorés")
print(f"Crops sauvés dans: {CROP_DIR}")

# Afficher quelques statistiques
if articles_db:
    original_sizes = [bbox[2] * bbox[3] for art in articles_db for bbox in [art["original_bbox"]]]
    improved_sizes = [bbox[2] * bbox[3] for art in articles_db for bbox in [art["improved_bbox"]]]
    
    print(f"Taille moyenne originale: {np.mean(original_sizes):.0f} px²")
    print(f"Taille moyenne améliorée: {np.mean(improved_sizes):.0f} px²")
    print(f"Augmentation moyenne: {np.mean(improved_sizes) / np.mean(original_sizes):.1f}x")