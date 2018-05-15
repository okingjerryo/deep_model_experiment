# coding=utf-8
"""根据搜索词下载百度图片"""
import re
import os, errno
import urllib.parse
import urllib.request
import time
import requests
from user_agent import generate_user_agent
from time import sleep
from tqdm import tqdm
import tensorflow as tf

elem_limit = 100


def download_page(url):
    """download raw content of the page

    Args:
        url (str): url of the page

    Returns:
        raw content of the page
    """
    try:
        headers = {}
        headers['User-Agent'] = generate_user_agent()
        headers['Referer'] = 'https://www.google.com'
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        return str(resp.read())
    except Exception as e:
        print('error while downloading page {0}'.format(url))
        print(e)
        return None


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_page(url):
    """parge the page and get all the links of images, max number is 100 due to limit by google

    Args:
        url (str): url of the page

    Returns:
        A set containing the urls of images
    """
    page_content = download_page(url)
    if page_content:
        link_list = re.findall('"ou":"(.*?)"', page_content)
        if len(link_list) == 0:
            print('get 0 links from page {0}'.format(url))
            return None
        else:
            return link_list
    else:
        return None


def get_onepage_urls(onepageurl, engine_type):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    page_content = None
    html = None
    if not onepageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        if engine_type == 'baidu':
            html = requests.get(onepageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = None
    fanye_url = None
    if engine_type == 'baidu':
        pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
        fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    if engine_type == 'google':
        pic_urls = parse_page(onepageurl)
        sleep(2)
    return pic_urls, fanye_url


def down_pic(pic_urls, keyword, typeword):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        string = None
        try:
            string_dir = "img/" + keyword + "/" + typeword + "/"
            mkdir_p(string_dir)
            tstamp = time.time()
            suffix = '.' + str.split(pic_url, ".")[-1]
            # print(suffix)
            string = string_dir + str(tstamp) + str(i + 1) + suffix
            pic = requests.get(pic_url, timeout=15)
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('成功下载' + keyword + " " + typeword + '第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print(keyword + " " + typeword + '第%s张图片下载失败: %s' % (str(i + 1), str(pic_url)))
            try:
                os.remove("./" + string)
            except:
                continue


def one_key_search(keyword, type, url):
    # 度
    url_init = None
    if url == 'baidu':
        if type == '原图':  # 限制搜索必须有人脸出现
            url_init_first = r'https://image.baidu.com/search/index?ct=201326592&z=2&tn=baiduimage&ipn=r&pn=0&istype=2&ie=utf-8&oe=utf-8&cl=2&lm=-1&st=-1&fr=&fmq=1525680097900_R&ic=0&se=&sme=&width=0&height=0&face=1&word='
            url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
        if type == '素描':  # 素描限制较为宽泛
            url_init_first = r'https://image.baidu.com/search/index?ct=201326592&z=2&tn=baiduimage&ipn=r&pn=0&istype=2&ie=utf-8&oe=utf-8&cl=2&lm=-1&st=-1&fr=&fmq=1525680097900_R&ic=0&se=&sme=&width=0&height=0&face=0&word='
            url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
    elif url == 'google':
        if type == '原图':
            search_query = (keyword).replace(' ', '%20')
            search_query = urllib.parse.quote(search_query.encode('utf-8'))
            print(search_query)
            url_init = r'https://www.google.com/search?q=' + search_query + '&source=lnms&tbm=isch'
        if type == '素描':
            search_query = (keyword + ' sketch').replace(' ', '%20')
            # search_query = urllib.parse.quote(search_query.encode('utf-8'))
            print(search_query)
            url_init = r'https://www.google.com/search?q=' + search_query + '&source=lnms&tbm=isch'

    onepage_urls, fanye_url = get_onepage_urls(url_init, url)
    onepage_urls = onepage_urls[:elem_limit]
    return onepage_urls


def one_engin_search(keyword, engien):
    all_pic_urls = []
    # 原图
    this_re = one_key_search(keyword, "原图", engien)
    print("原图" + str(len(this_re)))
    down_pic(list(set(this_re)), keyword, "原图")
    # 素描
    # this_re = one_key_search(keyword , "素描", engien)
    # print("素描"+str(len(this_re)))
    # down_pic(list(set(this_re)), keyword, "素描")


if __name__ == '__main__':
    # f = open('resource/globle_star.txt', 'r')
    # result = list()
    # for keyword in tqdm(open('resource/globle_star.txt'), ascii=True, desc='爬图像'):
    #     keyword = f.readline()[:-1]
    #     one_engin_search(keyword, 'google')
    one_engin_search("国内明星素描", 'google')
    one_engin_search("国内明星 素描", 'google')
