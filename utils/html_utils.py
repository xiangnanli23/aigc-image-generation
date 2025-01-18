
def generate_html(image_urls, labels, max_image_width=250, label_font_size=24):
    """
    生成包含图片和标签的HTML文件，所有图片横向排列，点击图片可放大显示，支持图片居中显示并通过键盘左右键切换图片。
    
    :param image_urls: 图片URL的列表
    :param labels: 对应图片标签的列表
    :param output_file: 输出HTML文件的名称，默认为'gallery.html'
    :param max_image_width: 图片的最大显示宽度，默认为150px
    :param label_font_size: 标签的字体大小，默认为14px
    """
    
    # 检查输入的图片URL和标签数量是否一致
    if len(image_urls) != len(labels):
        raise ValueError("The number of image URLs must match the number of labels.")
    
    # 创建HTML头部
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Gallery</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                margin: 0;
                padding: 20px;
            }}
            .gallery {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin-top: 20px;
            }}
            .gallery-item {{
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #fff;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                min-height: 250px; /* 保证图片和标签容器的最小高度相同 */
            }}
            .gallery-item:hover {{
                transform: scale(1.05);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            }}
            .gallery-item img {{
                max-width: {max_image_width}px;
                height: auto;
                border-radius: 8px;
                cursor: pointer;
                margin-bottom: 10px; /* 增加图片和标签之间的间距 */
            }}
            .gallery-item div {{
                font-size: {label_font_size}px;
                color: #333;
                text-align: center;
                width: 100%;
                padding: 5px;
                background-color: #f8f8f8;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .modal {{
                display: none;
                position: fixed;
                z-index: 1;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgba(0, 0, 0, 0.8);
                justify-content: center;
                align-items: center;
            }}
            .modal-content {{
                max-width: 90%;
                max-height: 90%;
                margin: auto;
                display: block;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }}
            .close {{
                position: absolute;
                top: 15px;
                right: 35px;
                color: #fff;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
                transition: color 0.2s ease;
            }}
            .close:hover {{
                color: #f00;
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center; color: #333;">Image Gallery</h1>
        <div class="gallery">
    '''
    
    # 添加每个图片及其标签到HTML内容中
    for url, label in zip(image_urls, labels):
        html_content += f'''
            <div class="gallery-item">
                <img src="{url}" alt="{label}" onclick="openModal('{url}')">
                <div>{label}</div>
            </div>
        '''
    
    # 创建模态框的HTML和JavaScript部分
    html_content += '''
        </div>
        
        <!-- The Modal -->
        <div id="myModal" class="modal">
            <span class="close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage" src="">
        </div>

        <script>
            var currentIndex = -1;  // 初始化为-1，确保初次加载时没有图片显示
            var images = [];
            
            // 将所有图片URL存储到数组中
            function initImages() {
                images = Array.from(document.querySelectorAll('.gallery-item img')).map(img => img.src);
            }
            
            // 打开模态框并显示图片
            function openModal(url) {
                var modal = document.getElementById("myModal");
                var modalImg = document.getElementById("modalImage");
                modal.style.display = "flex";
                modalImg.src = url;
                currentIndex = images.indexOf(url);
            }
            
            // 关闭模态框
            function closeModal() {
                var modal = document.getElementById("myModal");
                modal.style.display = "none";
                var modalImg = document.getElementById("modalImage");
                modalImg.src = ""; // 确保关闭时清除图片
                currentIndex = -1;  // 重置索引，确保下次打开时重新设置
            }
            
            // 切换图片
            function showImage(index) {
                if (index >= 0 && index < images.length) {
                    var modalImg = document.getElementById("modalImage");
                    modalImg.src = images[index];
                    currentIndex = index;
                }
            }
            
            // 键盘左右键控制图片切换
            document.addEventListener('keydown', function(event) {
                if (event.key === "Escape") {
                    closeModal();
                } else if (event.key === "ArrowLeft" && currentIndex > 0) {
                    showImage(currentIndex - 1);
                } else if (event.key === "ArrowRight" && currentIndex < images.length - 1) {
                    showImage(currentIndex + 1);
                }
            });
            
            // 初始化图片列表
            initImages();
        </script>
    </body>
    </html>
    '''
    return html_content