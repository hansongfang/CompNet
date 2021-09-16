def html_from_template(template_file, out_file, **kwargs):
    # print('load template file', template_file)
    with open(template_file, "r") as fin:
        lines = fin.readlines()
    template_string = "".join(lines)
    outfile_string = template_string.format(**kwargs)
    # print(f'Export html to {out_file}')
    with open(out_file, "w") as fout:
        fout.write(outfile_string)



