import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

def build_transform_pipeline(config):
    to_use_aug = config.get("to_use", False)
    aug_prob = config.get("aug_prob", 0.0)
    transforms_config = config.get("transforms", {})
    use_probs = config.get("to_use_funcs", {})
    transform_list = []

    transform_mapping = {
        "RandomRotation": lambda params: T.RandomRotation(**params),
        "RandomHorizontalFlip": lambda params: T.RandomHorizontalFlip(**params),
        "ColorJitter": lambda params: T.ColorJitter(**params),
        "Normalize": lambda params: T.Normalize(**params)
    }

    if to_use_aug:
        for trans_name, trans_details in transforms_config.items():
            params = trans_details.get("params", {})
            if trans_name in transform_mapping:
                t = transform_mapping[trans_name](params)
            else:
                raise ValueError(f"Unknown augmentation: {trans_name}")

            p = use_probs.get(trans_name, 0.0)
            if p:
                t = T.RandomApply([t], p=p)
                transform_list.append(t)

    return {"aug_prob": aug_prob, "transform_list": T.Compose(transform_list)}

def prepare_augmentation():
    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(0.4),
            v2.RandomVerticalFlip(0.1),
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
            v2.RandomApply(transforms=[v2.ColorJitter(brightness=0.3, hue=0.1)], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
    )

    transforms = v2.Compose(
        [
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(-5, 5), fill=1)], p=0.0),
            v2.RandomApply(transforms=[v2.Compose([
                v2.RandomResize(int(37 * 0.7), int(37 * 0.9)),
                v2.Resize(size=(37, 37))
            ])], p=0.0),
            v2.RandomApply(transforms=[v2.RandomPerspective(0.15, fill=1)], p=1.0)
            # v2.RandomApply(transforms=[v2.functional.perspective(startpoints=[[0, 0], [0, 37], [37, 37], [37, 0]],
            # 													 endpoints=[[0, 0], [0, 37], [uniRand(), 37], [uniRand(), 0]],
            # 													 fill=1)], p=1.0)
        ]
    )

    transforms = T.Compose([
        # Геометрические трансформации
        # T.RandomApply([T.RandomResizedCrop(100)], p=0.3),
        # T.RandomHorizontalFlip(p=0.4),  # Горизонтальное отражение
        # T.RandomApply([T.RandomAffine(
        #     degrees=10,  # Поворот в диапазоне ±10°
        #     translate=(0.1, 0.1),  # Сдвиг до 10% от размеров изображения
        #     scale=(0.9, 1.1),  # Изменение масштаба на ±10%
        #     shear=5,  # Сдвиг угла на ±5°
        #     interpolation=InterpolationMode.BILINEAR,  # Интерполяция
        #     fill=0)],  # Заполнение черным
        #     p=0.1
        # ),
        # Цветовые трансформации
        # T.RandomApply([T.ColorJitter(
        #     brightness=0.2,  # Изменение яркости на ±20%
        #     contrast=0.2,  # Изменение контраста на ±20%
        #     saturation=0.2,  # Изменение насыщенности на ±20%
        #     hue=0.05  # Изменение оттенка на ±5%
        # )], p=0.3),
        # Симуляция размытия
        # T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2)
        # Преобразование в тензор
        # T.ToTensor()
        # Нормализация
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms


