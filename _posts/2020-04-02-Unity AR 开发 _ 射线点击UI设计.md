---
layout: post
title:  "Unity AR 开发 | 射线点击UI设计"
summary: "射线点击UI设计"
author: Vinko
type_thumbnail: note.png
date: '2020-04-02 05:15:54  +0530'
category: jekyll
thumbnail: /assets/img/posts/hello.jpg
keywords: devlopr jekyll, how to use devlopr, devlopr, how to use devlopr-jekyll, devlopr-jekyll tutorial,best jekyll themes
permalink: /blog/rayuidesign/
usemathjax: true
---

@[TOC]
# Unity AR 开发 | 射线点击UI设计
## 1. 导入资源包

- Oculus Integration
- ZED_Unity_Plugin_v2.8.1

![test](https://imgur.com/415nqke)
<img src="https://imgur.com/415nqke" alt="Kitten"
title="test" width="150" height="100" />

## 2. 新建所需组件
 - Canvas
 	- Button
 	- Text
 - UIHelpers
 - ZED_Rig_Stereo
 - OVRCameraRig
**Remember: 停用LeftEyeAnchor, CenterEyeAnchor, RightEyeAnchor**
![Hierarchy列表](https://img-blog.csdnimg.cn/20200402041505770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg0MzIzNg==,size_16,color_FFFFFF,t_70#pic_center)

## 3. 修改Canvas属性
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200402043510931.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg0MzIzNg==,size_16,color_FFFFFF,t_70#pic_center)
### 3.1.调整Position
### 3.2.调整Scale
### 3.3.Render Mode选择 World Space
因为这样是AR-friendly。
### 3.4.设置Event Camera
这里选择`ZED_Rig_Stereo`下的`Left_eye`或者`Right_eye`都可以，拖拽操作。
### 3.5.关闭Graphic Raycaster
### 3.6.添加OVRRaycaster脚本
### 3.7.将UIHelper/LaserPointer拖拽为OVRRaycaster脚本下的Pointer变量

## 4. 修改UIHelper/LaserPointer属性
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200402044735471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg0MzIzNg==,size_16,color_FFFFFF,t_70#pic_center)
### 4.1.启用Line Renderer
### 4.2.修改Width>>0.005

## 5. 修改UIHelper/EventSystem属性

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200402045922572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg0MzIzNg==,size_16,color_FFFFFF,t_70#pic_center)
### 5.1.将`RightHandAnchor`或者`LeftHandAnchor`拖入RayTransform变量
### 5.2.设置JoyPadClickButton变量
设置确认按键。
| 属性 | 解释 |
|:---|:-------|
|**RayTransform**| 哪个手柄发射射线。默认为左手柄|
|**Cursor**| LaserPointer上的一个脚本，可以设置射线长度|
|**JoyPadClickButton**|设置哪个键触发UI事件|


## 6. 验证测试
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200402051037865.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg0MzIzNg==,size_16,color_FFFFFF,t_70#pic_center)
### 6.1.在Canvas/Button新建 InfoDisplay 脚本

```csharp
public class InfoDisplay : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

        
    }

    public void ShowDebug()
    {
        Debug.Log("Successful!!!点击成功了成功了");
    }
}

```

### 6.2.新建Click触发
### 6.3.执行ShowDebug函数
## 7. 应用
### 7.1.修改 InfoDisplay 脚本

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class InfoDisplay : MonoBehaviour
{
    public Text infoText;

    public string infoString;

    // Start is called before the first frame update
    void Start()
    {
        infoText = GameObject.Find("Text").GetComponent<Text>();
    }

    // Update is called once per frame
    void Update()
    {

        
    }

    public void ShowDebug()
    {
        infoText.text = infoString;
        //infoText.text = "";
    }
}
```
说明：用AR手柄点击该Button之后Text会显示相关说明

