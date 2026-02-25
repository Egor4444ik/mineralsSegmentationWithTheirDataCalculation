import os
import numpy as np
from vedo import *
from PIL import Image
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

from utils.settings import JPG_PATH, OBJ_PATH, WEIGHTS_PATH, RESULTS_PATH

class PointCloudObjectDetector:
    def __init__(self, obj_path = OBJ_PATH, jpg_path = JPG_PATH, model_path = WEIGHTS_PATH, results_path = RESULTS_PATH):
        """
        Инициализация детектора объектов в облаке точек
        
        Args:
            obj_path: путь к OBJ файлу с 3D моделью
            jpg_path: путь к текстуре JPG
            model_path: путь к модели YOLO
        """
        self.obj_path = obj_path
        self.jpg_path = jpg_path
        self.model_path = model_path
        self.mesh = None
        self.point_cloud = None  # Добавляем для хранения облака точек
        self.plotter = None
        self.screenshot_paths = []
        self.results_path = results_path
        self.all_boxes = []  # Список для хранения всех обнаруженных боксов
        self.camera_positions = []  # Список для хранения позиций камер
        self.extracted_points = []  # Список для хранения извлеченных точек

    def generate_report(self, all_results):
        """Генерация отчета о детекции"""
        report_path = f"{self.results_path}/combined/detection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("ОТЧЕТ О ДЕТЕКЦИИ ОБЪЕКТОВ\n")
            f.write("=" * 50 + "\n\n")
            
            total_objects = 0
            for view, data in all_results.items():
                num_objects = len(data['boxes'])
                total_objects += num_objects
                f.write(f"Ракурс: {view.upper()}\n")
                f.write(f"  Обнаружено объектов: {num_objects}\n")
                if num_objects > 0:
                    f.write(f"  Координаты боксов:\n")
                    for j, box in enumerate(data['boxes']):
                        f.write(f"    Объект {j+1}: x1={box[0]:.1f}, y1={box[1]:.1f}, "
                               f"x2={box[2]:.1f}, y2={box[3]:.1f}\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"ВСЕГО ОБНАРУЖЕНО ОБЪЕКТОВ: {total_objects}\n")
            f.write("=" * 50 + "\n")
        
        print(f"Отчет сохранен в: {report_path}")

    def check_files(self):
        """Проверка наличия всех необходимых файлов"""
        files_to_check = [
            (self.obj_path, "OBJ файл"),
            (self.jpg_path, "JPG файл"),
            (self.model_path, "модель YOLO")
        ]
        
        for file_path, file_desc in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Ошибка: {file_desc} не найден по пути: {file_path}")
        
        # Создаем папки для output
        os.makedirs("OutputData", exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(f"{self.results_path}/screenshots", exist_ok=True)
        os.makedirs(f"{self.results_path}/detections", exist_ok=True)
        os.makedirs(f"{self.results_path}/combined", exist_ok=True)
        os.makedirs(f"{self.results_path}/extracted_points", exist_ok=True)  # Новая папка для извлеченных точек

    def load_and_prepare_mesh(self):
        """Загрузка и подготовка 3D модели и облака точек"""
        # Загружаем mesh как обычно
        self.mesh = Mesh(self.obj_path).scale(10).pos(0, 0, 0)
        self.mesh.texture(self.jpg_path, scale=0.1)
        self.mesh.smooth(niter=20, pass_band=0.1, edge_angle=15, 
                        feature_angle=60, boundary=False)
        
        # Извлекаем точки из mesh для создания облака точек
        # Получаем вершины mesh
        vertices = self.mesh.vertices
        
        # Создаем облако точек из вершин mesh
        self.point_cloud = Points(vertices, r=5, c='gray', alpha=0.5)
        
        print(f"Загружено облако точек: {len(vertices)} точек")
        
        # Настройка окружения для рендеринга
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ["QT_API"] = "pyqt5"
        settings.default_backend = "vtk"

    def extract_points_in_bbox_3d(self, box_info):
        """
        Извлекает точки облака, попадающие в 3D bounding box
        
        Args:
            box_info: словарь с информацией о bounding box
            
        Returns:
            numpy array: точки внутри bounding box
        """
        view = box_info['view']
        box = box_info['box']
        
        bounds = self.mesh.bounds()
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        
        # Получаем все вершины mesh
        vertices = self.mesh.vertices()
        
        # Предполагаем размер изображения 800x600
        img_width, img_height = 800, 600
        
        # Конвертируем координаты box в относительные
        x1_rel = box[0] / img_width
        y1_rel = box[1] / img_height
        x2_rel = box[2] / img_width
        y2_rel = box[3] / img_height
        
        # Определяем границы в 3D в зависимости от ракурса
        if view in ['front', 'back']:
            z_pos = bounds[5] if view == 'front' else bounds[4]
            x_min = bounds[0] + x1_rel * x_range
            x_max = bounds[0] + x2_rel * x_range
            y_min = bounds[2] + y1_rel * y_range
            y_max = bounds[2] + y2_rel * y_range
            
            # Добавляем небольшую толщину по Z для захвата точек
            z_thickness = z_range * 0.1  # 10% от глубины модели
            z_min = z_pos - z_thickness/2
            z_max = z_pos + z_thickness/2
            
            # Фильтруем точки
            mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) & \
                   (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) & \
                   (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                   
        elif view in ['left', 'right']:
            x_pos = bounds[0] if view == 'left' else bounds[1]
            y_min = bounds[2] + y1_rel * y_range
            y_max = bounds[2] + y2_rel * y_range
            z_min = bounds[4] + x1_rel * z_range
            z_max = bounds[4] + x2_rel * z_range
            
            # Добавляем толщину по X
            x_thickness = x_range * 0.1
            x_min = x_pos - x_thickness/2
            x_max = x_pos + x_thickness/2
            
            mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) & \
                   (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) & \
                   (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                   
        elif view in ['top', 'bottom']:
            y_pos = bounds[3] if view == 'top' else bounds[2]
            x_min = bounds[0] + x1_rel * x_range
            x_max = bounds[0] + x2_rel * x_range
            z_min = bounds[4] + y1_rel * z_range
            z_max = bounds[4] + y2_rel * z_range
            
            # Добавляем толщину по Y
            y_thickness = y_range * 0.1
            y_min = y_pos - y_thickness/2
            y_max = y_pos + y_thickness/2
            
            mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) & \
                   (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) & \
                   (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
        extracted = vertices[mask]
        return extracted, mask
    
    def extract_all_points_in_bboxes(self):
        """Извлекает точки для всех обнаруженных bounding boxes"""
        self.extracted_points = []
        all_points_list = []
        for i, box_info in enumerate(self.all_boxes):
            extracted_points, mask = self.extract_points_in_bbox_3d(box_info)
            if len(extracted_points) > 0:
                point_info = {
                    'index': i,
                    'view': box_info['view'],
                    'box': box_info['box'],
                    'points': extracted_points,
                    'mask': mask,
                    'count': len(extracted_points)
                }
                self.extracted_points.append(point_info)
                all_points_list.append(extracted_points)
                # Сохраняем извлеченные точки в файл
                points_path = f"{self.results_path}/extracted_points/bbox_{i+1}_{box_info['view']}.txt"
                np.savetxt(points_path, extracted_points, header='x y z', comments='')
                # Также сохраняем в формате PLY для визуализации
                self.save_points_as_ply(extracted_points, 
                    f"{self.results_path}/extracted_points/bbox_{i+1}_{box_info['view']}.ply")
                print(f"Извлечено {len(extracted_points)} точек из {box_info['view']} bbox {i+1}")

        # --- Пересечение всех облаков точек ---
        intersection_points = None
        if all_points_list:
            sets = [set(map(tuple, pts)) for pts in all_points_list if len(pts) > 0]
            if sets:
                intersection_points = set.intersection(*sets)
                intersection_points = np.array(list(intersection_points)) if intersection_points else np.empty((0, 3))
                # Сохраняем пересечение
                intersection_txt = f"{self.results_path}/extracted_points/intersection_points.txt"
                intersection_ply = f"{self.results_path}/extracted_points/intersection_points.ply"
                np.savetxt(intersection_txt, intersection_points, header='x y z', comments='')
                self.save_points_as_ply(intersection_points, intersection_ply)
                print(f"Сохранено {len(intersection_points)} точек-пересечений во всех bbox: {intersection_txt}")
                # Визуализация пересечения
                self.visualize_intersection_points(intersection_points)
        return self.extracted_points

    def visualize_intersection_points(self, intersection_points):
        """Визуализирует пересечение точек bounding boxes"""
        if intersection_points is None or len(intersection_points) == 0:
            print("Нет точек-пересечений для визуализации")
            return
        plotter = Plotter()
        self.mesh.alpha(0.3)
        plotter.add(self.mesh)
        points_cloud = Points(intersection_points, r=5, c='magenta', alpha=0.9)
        plotter.add(points_cloud)
        center = np.mean(intersection_points, axis=0)
        text = Text3D(f"Intersection ({len(intersection_points)})", center + [0, 0, 5], s=3, c='magenta')
        plotter.add(text)
        plotter.add(Axes(self.mesh))
        print("Показываю визуализацию пересечения точек bounding boxes...")
        plotter.show(title="Пересечение точек bounding boxes", viewup="y", axes=1).close()
    
    def save_points_as_ply(self, points, filename):
        """Сохраняет точки в формате PLY"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

    def visualize_extracted_points(self):
        """Визуализирует извлеченные точки разными цветами"""
        if not self.extracted_points:
            print("Нет извлеченных точек для визуализации")
            return
        
        plotter = Plotter()
        
        # Показываем исходный mesh полупрозрачным
        self.mesh.alpha(0.3)
        plotter.add(self.mesh)
        
        # Разные цвета для разных bounding boxes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, point_info in enumerate(self.extracted_points):
            color = colors[i % len(colors)]
            
            # Создаем облако точек для извлеченных точек
            points_cloud = Points(point_info['points'], r=3, c=color, alpha=0.8)
            plotter.add(points_cloud)
            
            # Добавляем текст с информацией
            center = np.mean(point_info['points'], axis=0)
            text = Text3D(f"Object {i+1} ({point_info['view']})", 
                         center + [0, 0, 5], s=3, c=color)
            plotter.add(text)
        
        plotter.add(Axes(self.mesh))
        print("Показываю визуализацию извлеченных точек...")
        plotter.show(title="Извлеченные точки из bounding boxes", viewup="y", axes=1).close()

    def analyze_extracted_points(self):
        """Анализирует статистику извлеченных точек"""
        if not self.extracted_points:
            print("Нет извлеченных точек для анализа")
            return
        
        report_path = f"{self.results_path}/extracted_points/points_analysis.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("АНАЛИЗ ИЗВЛЕЧЕННЫХ ТОЧЕК ИЗ BOUNDING BOXES\n")
            f.write("=" * 60 + "\n\n")
            
            total_points = 0
            for i, point_info in enumerate(self.extracted_points):
                points = point_info['points']
                total_points += len(points)
                
                f.write(f"Объект {i+1} (ракурс: {point_info['view']}):\n")
                f.write(f"  Количество точек: {len(points)}\n")
                f.write(f"  Центр: ({np.mean(points[:,0]):.2f}, {np.mean(points[:,1]):.2f}, {np.mean(points[:,2]):.2f})\n")
                f.write(f"  Размеры (x,y,z): ({np.max(points[:,0])-np.min(points[:,0]):.2f}, "
                       f"{np.max(points[:,1])-np.min(points[:,1]):.2f}, "
                       f"{np.max(points[:,2])-np.min(points[:,2]):.2f})\n\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"ВСЕГО ИЗВЛЕЧЕНО ТОЧЕК: {total_points}\n")
            f.write("=" * 60 + "\n")
        
        print(f"Анализ точек сохранен в: {report_path}")

    def run_pipeline(self, conf=0.01, iou=0.001):
        """
        Запуск полного пайплайна обработки с 6 ракурсами и извлечением точек
        """
        try:
            # 1. Проверка файлов
            self.check_files()
            
            # 2. Загрузка модели и облака точек
            self.load_and_prepare_mesh()
            
            # 3. Создание 6 скриншотов
            self.capture_multiple_screenshots()
            
            # 4. Детекция объектов на всех скриншотах
            all_results = self.detect_objects_on_all_screenshots(conf, iou)
            
            # 5. Создание составного изображения
            self.create_composite_image(all_results)
            
            # 6. Генерация отчета о детекции
            self.generate_report(all_results)
            
            # 7. Извлечение точек из bounding boxes (НОВОЕ!)
            if len(self.all_boxes) > 0:
                print("\n" + "="*50)
                print("ИЗВЛЕЧЕНИЕ ТОЧЕК ИЗ BOUNDING BOXES")
                print("="*50)
                
                # Извлекаем точки
                self.extract_all_points_in_bboxes()
                
                # Анализируем извлеченные точки
                self.analyze_extracted_points()
                
                # Визуализируем извлеченные точки
                self.visualize_extracted_points()
                
                # Обычная визуализация со всеми детекциями
                self.visualize_3d_with_all_detections()
            else:
                print("Объекты не обнаружены ни на одном ракурсе.")
                plotter = Plotter()
                plotter.add(self.mesh)
                plotter.add(Axes(self.mesh))
                plotter.show(title="Модель без обнаруженных объектов", viewup="y").close()
                
        except Exception as e:
            print(f"Ошибка в пайплайне: {e}")
            raise
        
    def setup_camera_positions(self):
        """Настройка 6 позиций камеры для обзора со всех сторон"""
        bounds = self.mesh.bounds()
        center = [(bounds[0] + bounds[1])/2, 
                 (bounds[2] + bounds[3])/2, 
                 (bounds[4] + bounds[5])/2]
        
        # Рассчитываем размер модели для расстояния камеры
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]
        max_size = max(size_x, size_y, size_z)
        distance = max_size * 3  # Расстояние камеры от центра
        
        # 6 позиций камеры (вокруг объекта)
        positions = [
            # Спереди
            ([center[0], center[1] + distance/2, center[2]], [0, 1, 0]),
            # Сзади
            ([center[0], center[1] - distance/2, center[2]], [0, 1, 0]),
            # Слева
            ([center[0] - distance/2, center[1], center[2]], [0, 1, 0]),
            # Справа
            ([center[0] + distance/2, center[1], center[2]], [0, 1, 0]),
            # Сверху
            ([center[0], center[1], center[2] + distance/2], [1, 0, 0]),
            # Снизу
            ([center[0], center[1], center[2] - distance/2], [0, 0, 1])
        ]
        
        return positions, center
    
    def capture_multiple_screenshots(self):
        """Создание 6 скриншотов с разных сторон"""
        positions, center = self.setup_camera_positions()
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        self.screenshot_paths = []
        
        for i, (position, view_up) in enumerate(positions):
            # Создаем новый плоттер для каждого ракурса
            plotter = Plotter(offscreen=True)
            plotter.add(self.mesh)
            
            # Настраиваем камеру
            plotter.camera.SetPosition(position)
            plotter.camera.SetFocalPoint(center)
            plotter.camera.SetViewUp(view_up)
            plotter.reset_camera()
            plotter.background('white')
            
            # Сохраняем скриншот
            screenshot_path = f"{self.results_path}/screenshots/{view_names[i]}.png"
            
            try:
                plotter.screenshot(filename=screenshot_path)
                print(f"Скриншот {view_names[i]} сохранен в: {screenshot_path}")
                
                # Конвертация в RGB
                img = Image.open(screenshot_path).convert("RGB")
                img.save(screenshot_path)
                
                self.screenshot_paths.append({
                    'path': screenshot_path,
                    'view': view_names[i],
                    'position': position,
                    'view_up': view_up
                })
                
            except Exception as e:
                print(f"Ошибка при создании скриншота {view_names[i]}: {e}")
            finally:
                plotter.close()
        
        return self.screenshot_paths
    
    def detect_objects_on_all_screenshots(self, conf=0.01, iou=0.001):
        """
        Детекция объектов на всех скриншотах
        
        Returns:
            all_results: словарь с результатами для каждого ракурса
        """
        model = YOLO(self.model_path)
        all_results = {}
        self.all_boxes = []
        
        for screenshot_info in self.screenshot_paths:
            view = screenshot_info['view']
            path = screenshot_info['path']
            
            # Детекция
            results = model(path, conf=conf, iou=iou, verbose=False)
            
            # Получаем bounding boxes
            boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Сохраняем информацию о боксах с меткой ракурса
                for box in boxes:
                    self.all_boxes.append({
                        'view': view,
                        'box': box,
                        'position': screenshot_info['position'],
                        'view_up': screenshot_info['view_up']
                    })
            
            # Сохраняем результат с размеченными объектами
            if len(results) > 0:
                plotted_image = results[0].plot()
                result_path = f"{self.results_path}/detections/{view}_detected.png"
                cv2.imwrite(result_path, plotted_image)
                print(f"Результат детекции для {view} сохранен в: {result_path}")
                
                # Конвертируем BGR в RGB для отображения
                plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
                
                all_results[view] = {
                    'results': results,
                    'boxes': boxes,
                    'image': plotted_image_rgb,
                    'path': result_path
                }
            
        return all_results
    
    def create_composite_image(self, all_results):
        """Создание составного изображения со всеми ракурсами"""
        if not all_results:
            return
        
        # Создаем фигуру 2x3 для 6 изображений
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        views_order = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        for i, view in enumerate(views_order):
            if view in all_results:
                # Отображаем изображение
                axes[i].imshow(all_results[view]['image'])
                axes[i].set_title(f'{view.upper()} view')
                axes[i].axis('off')
                
                # Добавляем информацию о количестве объектов
                num_objects = len(all_results[view]['boxes'])
                axes[i].text(10, 30, f'Objects: {num_objects}', 
                           color='red', fontsize=12, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7))
            else:
                axes[i].text(0.5, 0.5, f'{view} - No data', 
                           ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        composite_path = f"{self.results_path}/combined/all_views_composite.png"
        plt.savefig(composite_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Составное изображение сохранено в: {composite_path}")
        
        # Также сохраняем отдельно изображение с детекциями в 3D
        self.create_3d_detection_summary()
    
    def create_3d_detection_summary(self):
        """Создание 3D сводки со всеми обнаруженными областями"""
        import numpy as np
        plotter_3d = Plotter(offscreen=True)
        plotter_3d.add(self.mesh)
        
        bounds = self.mesh.bounds()
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        
        # Разные цвета для разных ракурсов
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Для каждого обнаруженного бокса
        for i, box_info in enumerate(self.all_boxes):
            view = box_info['view']
            box = box_info['box']
            color = colors[i % len(colors)]
            
            # Предполагаем размер изображения 800x600
            img_width, img_height = 800, 600
            
            # Конвертируем координаты box в относительные
            x1_rel = box[0] / img_width
            y1_rel = box[1] / img_height
            x2_rel = box[2] / img_width
            y2_rel = box[3] / img_height
            
            # В зависимости от ракурса, проецируем на соответствующую плоскость
            # Формируем points в зависимости от view
            if view in ['front', 'back']:
                z_pos = bounds[5] if view == 'front' else bounds[4]
                x_min = bounds[0] + x1_rel * x_range
                x_max = bounds[0] + x2_rel * x_range
                y_min = bounds[2] + y1_rel * y_range
                y_max = bounds[2] + y2_rel * y_range
                points = [
                    [x_min, y_min, z_pos],
                    [x_max, y_min, z_pos],
                    [x_max, y_max, z_pos],
                    [x_min, y_max, z_pos],
                    [x_min, y_min, z_pos]
                ]
            elif view in ['left', 'right']:
                x_pos = bounds[0] if view == 'left' else bounds[1]
                y_min = bounds[2] + y1_rel * y_range
                y_max = bounds[2] + y2_rel * y_range
                z_min = bounds[4] + x1_rel * z_range
                z_max = bounds[4] + x2_rel * z_range
                points = [
                    [x_pos, y_min, z_min],
                    [x_pos, y_max, z_min],
                    [x_pos, y_max, z_max],
                    [x_pos, y_min, z_max],
                    [x_pos, y_min, z_min]
                ]
            elif view in ['top', 'bottom']:
                y_pos = bounds[3] if view == 'top' else bounds[2]
                x_min = bounds[0] + x1_rel * x_range
                x_max = bounds[0] + x2_rel * x_range
                z_min = bounds[4] + y1_rel * z_range
                z_max = bounds[4] + y2_rel * z_range
                points = [
                    [x_min, y_pos, z_min],
                    [x_max, y_pos, z_min],
                    [x_max, y_pos, z_max],
                    [x_min, y_pos, z_max],
                    [x_min, y_pos, z_min]
                ]
            else:
                print(f"[WARNING] Неизвестный view '{view}' для бокса {i+1}. Пропуск.")
                continue

            # Проверяем, что каждый элемент — это список длины 3
            valid_points = all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 3 for pt in points)
            if not valid_points:
                print(f"[WARNING] Некорректные точки для бокса {i+1} ({view}): {points}. Пропуск.")
                continue

            points_list = points
            # Проверка: points_list должен быть списком списков длины 3
            if not (isinstance(points_list, list) and all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 3 for pt in points_list)):
                print(f"[WARNING] Некорректные точки для бокса {i+1} ({view}): {points_list}. Пропуск.")
                continue
            if not np.allclose(points_list[0], points_list[-1]):
                points_list.append(points_list[0])
            # Преобразуем все координаты к float
            import numpy as np
            points_list_clean = [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in points_list]
            print(f"[DEBUG] points_list для бокса {i+1} ({view}): {points_list_clean}")
            try:
                bbox_lines = Lines(points_list_clean).c(color).lw(2)
                plotter_3d.add(bbox_lines)
            except Exception as e:
                print(f"[ERROR] Не удалось построить Lines для бокса {i+1} ({view}): {points_list_clean}. Ошибка: {e}")
                continue

            # Добавляем текст с номером объекта и ракурсом
            text_pos = points[0]  # Используем первую точку для текста
            text = Text3D(f"{view[:2]}{i+1}", text_pos, s=5, c=color)
            plotter_3d.add(text)
        
        # Добавляем оси и настраиваем вид
        plotter_3d.add(Axes(self.mesh))
        
        # Сохраняем 3D сводку
        summary_path = f"{self.results_path}/combined/3d_detection_summary.png"
        
        # Делаем несколько ракурсов для 3D сводки
        center = [(bounds[0] + bounds[1])/2, 
                 (bounds[2] + bounds[3])/2, 
                 (bounds[4] + bounds[5])/2]
        
        # Ракурс 1: изометрический вид
        plotter_3d.camera.SetPosition([center[0] + 100, center[1] + 100, center[2] + 50])
        plotter_3d.camera.SetFocalPoint(center)
        plotter_3d.screenshot(filename=summary_path.replace('.png', '_iso.png'))
        
        # Ракурс 2: вид спереди
        plotter_3d.camera.SetPosition([center[0], center[1] + 150, center[2]])
        plotter_3d.camera.SetFocalPoint(center)
        plotter_3d.screenshot(filename=summary_path.replace('.png', '_front.png'))
        
        print(f"3D сводки сохранены в: {self.results_path}/combined/")
        plotter_3d.close()
    
    def visualize_3d_with_all_detections(self):
        """Интерактивная 3D визуализация со всеми обнаруженными объектами"""
        plotter = Plotter()
        plotter.add(self.mesh)
        
        bounds = self.mesh.bounds()
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        
        # Разные цвета для разных ракурсов
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, box_info in enumerate(self.all_boxes):
            view = box_info['view']
            box = box_info['box']
            color = colors[i % len(colors)]
            
            img_width, img_height = 800, 600
            
            x1_rel = box[0] / img_width
            y1_rel = box[1] / img_height
            x2_rel = box[2] / img_width
            y2_rel = box[3] / img_height
            
            # Проецируем на соответствующую плоскость
            if view in ['front', 'back']:
                z_pos = bounds[5] if view == 'front' else bounds[4]
                x_min = bounds[0] + x1_rel * x_range
                x_max = bounds[0] + x2_rel * x_range
                y_min = bounds[2] + y1_rel * y_range
                y_max = bounds[2] + y2_rel * y_range
                
                # Создаем плоскость с проекцией
                plane = Plane(pos=[(x_min+x_max)/2, (y_min+y_max)/2, z_pos],
                            normal=[0, 0, 1 if view == 'front' else -1],
                            s=(x_max-x_min, y_max-y_min)).c(color).alpha(0.3)
                plotter.add(plane)
                
                # Добавляем рамку
                points = [
                    [x_min, y_min, z_pos],
                    [x_max, y_min, z_pos],
                    [x_max, y_max, z_pos],
                    [x_min, y_max, z_pos],
                    [x_min, y_min, z_pos]
                ]
                bbox_lines = Lines(points, closed=True).c(color).lw(3)
                plotter.add(bbox_lines)
                
            elif view in ['left', 'right']:
                x_pos = bounds[0] if view == 'left' else bounds[1]
                y_min = bounds[2] + y1_rel * y_range
                y_max = bounds[2] + y2_rel * y_range
                z_min = bounds[4] + x1_rel * z_range
                z_max = bounds[4] + x2_rel * z_range
                
                plane = Plane(pos=[x_pos, (y_min+y_max)/2, (z_min+z_max)/2],
                            normal=[1 if view == 'left' else -1, 0, 0],
                            s=(y_max-y_min, z_max-z_min)).c(color).alpha(0.3)
                plotter.add(plane)
                
                points = [
                    [x_pos, y_min, z_min],
                    [x_pos, y_max, z_min],
                    [x_pos, y_max, z_max],
                    [x_pos, y_min, z_max],
                    [x_pos, y_min, z_min]
                ]
                bbox_lines = Lines(points, closed=True).c(color).lw(3)
                plotter.add(bbox_lines)
                
            elif view in ['top', 'bottom']:
                y_pos = bounds[3] if view == 'top' else bounds[2]
                x_min = bounds[0] + x1_rel * x_range
                x_max = bounds[0] + x2_rel * x_range
                z_min = bounds[4] + y1_rel * z_range
                z_max = bounds[4] + y2_rel * z_range
                
                plane = Plane(pos=[(x_min+x_max)/2, y_pos, (z_min+z_max)/2],
                            normal=[0, 1 if view == 'top' else -1, 0],
                            s=(x_max-x_min, z_max-z_min)).c(color).alpha(0.3)
                plotter.add(plane)
                
                points = [
                    [x_min, y_pos, z_min],
                    [x_max, y_pos, z_min],
                    [x_max, y_pos, z_max],
                    [x_min, y_pos, z_max],
                    [x_min, y_pos, z_min]
                ]
                bbox_lines = Lines(points, closed=True).c(color).lw(3)
                plotter.add(bbox_lines)
        
        plotter.add(Axes(self.mesh))
        print("Показываю интерактивную 3D визуализацию со всеми обнаруженными объектами...")
        plotter.show(title="Все обнаруженные объекты в 3D", viewup="y", axes=1).close()

def process_pointcloud_with_detection(obj_path = OBJ_PATH, jpg_path = JPG_PATH, model_path = WEIGHTS_PATH, conf=0.01, iou=0.001):
    """
    Основная функция для обработки облака точек с детекцией объектов с 6 сторон
    и извлечением точек из bounding boxes
    """
    detector = PointCloudObjectDetector(obj_path, jpg_path, model_path)
    detector.run_pipeline(conf, iou)

if __name__ == "__main__":
    process_pointcloud_with_detection()