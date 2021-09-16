from bs4 import BeautifulSoup
from CompNet.utils.format_tool import html_from_template


def get_canvases_tag_from_img_list(img_list, loss_list):
    soup = BeautifulSoup("<div id='canvases'></div>", features="html.parser")
    original_tag = soup.div

    num_canvas = len(img_list)
    for i in range(num_canvas):
        slide_tag = soup.new_tag("div")
        slide_tag["class"] = "slide"
        canvas_tag = soup.new_tag('canvas', id=f"canvas{i+1}")
        slide_tag.append(canvas_tag)
        figure_tag = soup.new_tag('figure')
        figure_tag["class"] = "image is-128x128"
        for img in img_list[i]:
            img_tag = soup.new_tag('img', src=img, alt='img')
            figure_tag.append(img_tag)
        slide_tag.append(figure_tag)
        loss_tag = soup.new_tag('h5')
        loss_tag.string=f'loss={loss_list[i]:.3f}'
        slide_tag.append(loss_tag)
        original_tag.append(slide_tag)

    return original_tag.prettify()


def get_ncanvas_html(img_list, ply_list, ply_name, obj_name, model_name, template_file, out_file, loss_list=None):
    # generate
    if loss_list is None:
        loss_list = [0.0] * len(img_list)

    canvases_ss = get_canvases_tag_from_img_list(img_list, loss_list)
    file_ss = [f"\'{item}\'" for item in ply_list]
    file_ss = ",".join(file_ss)
    file_name_ss = ",".join(ply_name)

    html_from_template(template_file, out_file,
                       canvases_string=canvases_ss,
                       obj_name=obj_name,
                       model_name=model_name,
                       ply_list=file_ss,
                       ply_name_list=file_name_ss)


if __name__ == "__main__":
    img_list = [['img.png', 'mask.png']]
    loss_list= [0.51231231]
    canvases_ss = get_canvases_tag_from_img_list(img_list, loss_list)
    print(canvases_ss)