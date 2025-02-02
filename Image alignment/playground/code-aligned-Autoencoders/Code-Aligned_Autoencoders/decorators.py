import tensorflow as tf
from functools import wraps
from timeit import default_timer as timer


def timed(func):
    """ Times function call """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timer()
        out = func(*args, **kwargs)
        stop = timer()

        return stop - start, out

    return wrapper


def _change_image_range(tensor):
    """ Take image to [0, 1] """
    return (tensor - tf.reduce_min(tensor)) / (
        tf.reduce_max(tensor) - tf.reduce_min(tensor)
    )


def write_image_to_summary(image, writer, name, pre_process=None):
    if image.dtype == tf.bool:
        image = tf.cast(image, tf.float32)

    image = _change_image_range(image)
    if pre_process is not None:
        image = pre_process(image)

    # device is a workaround for github.com/tensorflow/tensorflow/issues/28007
    with tf.device("cpu:0"):
        with writer.as_default():
            tf.summary.image(name, image)


# def write_image_to_png(image, filename):
#     """ Write [0, 1] image to png file """
#     if tf.rank(image) == 4:
#         image = image[0]
#     image = _change_image_range(image)
#     image = tf.cast(255 * image, tf.uint8)  # unit8
#     # # 转换为灰度图像
#     # grayscale_image = tf.image.rgb_to_grayscale(image)
#     # # 将图像数据类型更改为适合保存的类型
#     # grayscale_image = tf.image.convert_image_dtype(grayscale_image, tf.uint8)
#     contents = tf.image.encode_png(image)
#     tf.io.write_file(filename, contents)

def write_image_to_png(image, filename):
    """ Write [0, 1] image to png file """
    if tf.rank(image) == 4:
        image = image[0]
    image = _change_image_range(image)
    # img_batch = tf.expand_dims(image, axis=0)
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    # 将图像转换为灰度
    grayscale_image = tf.image.rgb_to_grayscale(image)
    # 调整像素范围到 [0, 255]
    grayscale_image = tf.cast(255 * grayscale_image, tf.uint8)    # uint8
    # 编码为 PNG 格式
    contents = tf.image.encode_png(grayscale_image)
    # 写入文件
    tf.io.write_file(filename, contents)


def image_to_tensorboard(static_name=None, pre_process=None):
    """
        Create decorator to write function output with tf.summary.image.
        Wrapped function should return
            image - (batch_size, h, w)

        TensorBoard tag 'name' can be provided at decoration time as
        'static_name' or as a keyword-only argument 'name' at call time.
        If neither are provided, the decorator does nothing.

        Assumes tf.summary.experimental.get_step() is not None
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, name=None, **kwargs):
            name = name if name is not None else static_name
            out = tmp = func(self, *args, **kwargs)
            if tmp.shape[-1] > 3:
                if "x" in name:
                    ch = self.channels["x"]
                    name += str(self.channels["x"])
                elif "y" in name:
                    ch = self.channels["y"]
                    name += str(self.channels["y"])
                tmp2 = tf.concat(
                    [
                        tf.expand_dims(tmp[..., ch[0]], -1),
                        tf.expand_dims(tmp[..., ch[1]], -1),
                        tf.expand_dims(tmp[..., ch[2]], -1),
                    ],
                    3,
                )
            else:
                tmp2 = tmp
            # new_name = "aaa"  # 你可以修改成任何你想要的新名称
            if (
                name is not None
                and self.evaluation_frequency > 0
                and not tf.cast(
                    tf.summary.experimental.get_step() % self.evaluation_frequency,
                    dtype=tf.bool,
                )
            ) or self._save_images:
                write_image_to_summary(tmp2, self.tb_writer, name, pre_process)
            if self._save_images and name is not None:
                filename = self._image_dir + tf.constant(f"/{name}.png")
                write_image_to_png(tmp2, filename)
                if name in ['x_code', 'x_hat', 'x_tilde', 'y_code', 'y_hat', 'y_tilde']:
                    new_filename = tf.constant("E:/data-alignment/playground/playground-result") + tf.constant(f"/{name}.png")
                    write_image_to_png(tmp2, new_filename)

            return out

        return wrapper

    return decorator
