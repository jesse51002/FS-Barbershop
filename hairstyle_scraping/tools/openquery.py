import webbrowser


MODES = {
    "face": "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&tbs=isz:lt,islt:svga,itp:face",
    "upper": "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&tbs=isz:lt,islt:xga,itp:photo,ic:color"
}

mode = "upper"


query = "afro hairstyle waist up picture"
webbrowser.open(MODES[mode]%(query))