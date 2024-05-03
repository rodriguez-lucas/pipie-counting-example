from ultralytics import RTDETR


def main():
    model = RTDETR('rtdetr-l.pt')
    model.train(data='./dataset/data.yaml', epochs=1, imgsz=640, project='pipe-counting', name='train')
    # results = model.predict('bus.jpg', save=True, project='pipe-counting', name='predict')


if __name__ == '__main__':
    main()
