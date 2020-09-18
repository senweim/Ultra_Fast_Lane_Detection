#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def log_out(str_out, f_out):
    f_out.write(str_out + '\n')
    f_out.flush()
    print(str_out)

