import dataset
import args
import model
import tensorflow as tf


# 主函数中调用estimator的接口
def get_estimator(feature_columns):
    return tf.estimator.Estimator(
        model_fn=model.cGAN_main,
        params={
            'feature': feature_columns
        }
    )


def main():
    pass


if __name__ == '__main__':
    main()
