---
layout: post
title:  "Unity AR开发 | 获取Controller的Position和Rotation"
summary: "射线点击UI设计"
author: Vinko
type: notebook
date: '2020-04-03 04:44:45  +0530'
category: jekyll
thumbnail: /assets/img/posts/unity_arui_controller.jpg
keywords: devlopr jekyll, how to use devlopr, devlopr, how to use devlopr-jekyll, devlopr-jekyll tutorial,best jekyll themes
permalink: /blog/2020-04-03-unity-controller-pose/
usemathjax: true
---


# Unity AR开发 | 获取Controller的Position和Rotation


## 1. 前序
* Unity场景搭建：[射线点击UI设计](https://blog.csdn.net/weixin_45843236/article/details/105260464)
* 文章目的：当按下Controller某个按键时，记录手柄在空间中的位姿（位置和姿态）

## 2. 新建所需组件
### 2.1 在Canvas目录下新建两个Button
![新建两个Button](https://i.imgur.com/FOnMoBM.png)
* 点击<kbd>Button-2</kbd>用于记录**Position**
* 点击<kbd>Button-3</kbd>用于记录**Rotation**

## 3. 设置Button-2 属性脚本
### 3.1 新建ShowPosition脚本并添加至Button-2
```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class ShowPosition : MonoBehaviour
{
    public Text infoText;
    // Start is called before the first frame update
    void Start()
    {
        infoText = GameObject.Find("Text").GetComponent<Text>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void positionShow()
    {
        Vector3 position = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
        Debug.Log(position);
        infoText.text = position.ToString();
    }
}

```
### 3.2 定义触发功能
将Button-2拖入触发物体框中，函数选择ShowPosition.positionShow, 意思是调用Button-2【GameObject】下的ShowPosition【Component】里的positionShow【函数】。
![定义触发功能](https://i.imgur.com/RqbUh5N.png)
## 4. 设置Button-3 属性脚本
### 4.1 新建ShowRotation脚本并添加至Button-3

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowRotation : MonoBehaviour
{
    public Text infoText;
    // Start is called before the first frame update
    void Start()
    {
        infoText = GameObject.Find("Text").GetComponent<Text>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void rotationShow()
    {
        Quaternion rotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);
        
        infoText.text = rotation.ToString();
    }
}

```
### 4.2 定义触发功能
类似3.2

## 5. 运行结果
![result01](https://i.imgur.com/kjMDE28.png)
![result02](https://i.imgur.com/pn2eZTI.png)
***
## 6. 总结
### 6.1 主要调用了OVRInput里面的Get()系列函数

> GetLocalControllerPosition
> 
和
> GetLocalControllerRotation
> 
分别返回一个**Vector3** 和 **Quaaternion**
使用句型如下：

`Vector3 position = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);`
`Quaternion rotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);`
**OVRInput.Controller.RTouch**代表右手柄
### 6.2 主要Button组件里面脚本要调用的函数设置为Public，否则无法在触发设置里找到
### 6.3 text的显示只能是字符，要用tostring函数转换
使用句型如下：
`infoText.text = rotation.ToString();`

