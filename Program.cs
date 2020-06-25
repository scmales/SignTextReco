using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SignTextReco
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"..\..\images"; //image folder
            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] files = root.GetFiles();
            var signReco = new SignReco();
            
            foreach (var file in files)
            {
                var filename = file.FullName;
                Mat img = new Mat(filename);
                Bitmap image = new Bitmap(filename);
                //Mat img = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);

                // 设置不需要文字识别的区域，不同图片需要修改
                var bounds = new List<System.Windows.Rect>{new System.Windows.Rect(0, 0, 1024, 408), new System.Windows.Rect(1200, 0, 670, 408)} ; 
                var textDic = TextReco.GetTextPosition(image, bounds);
                //输出文字和位置
                foreach(var pair in textDic)
                {
                    Console.WriteLine(pair.Key + "  " + pair.Value.ToString());

                }

                //画出检测出的加号
                var resPlusList = signReco.GetPlusSignPostion(image);
                for (int i = 0; i < resPlusList.Count; i++)
                {
                    var res = new OpenCvSharp.Point(resPlusList[i].X, resPlusList[i].Y);
                    Rect r = new Rect(new OpenCvSharp.Point(res.X - 10, res.Y - 10), new OpenCvSharp.Size(20, 20));
                    Cv2.Rectangle(img, r, Scalar.LimeGreen, 3);
                }
                //画出检测出的箭头
                var resList = signReco.GetArrowsSignPostion(image);
                for (int i = 0; i < resList.Count; i++)
                {
                    var start = new OpenCvSharp.Point(resList[i].Key[0].X, resList[i].Key[0].Y);
                    var end = new OpenCvSharp.Point(resList[i].Key[1].X, resList[i].Key[1].Y);
                    if (resList[i].Value == 1)
                    {
                        Cv2.Line(img, start, end, Scalar.Red, 2);

                    }
                    else if (start.X > end.X && resList[i].Value == 0)
                        Cv2.Line(img, start, end, Scalar.Blue, 2);
                    else if (start.X < end.X && resList[i].Value == 0)
                        Cv2.Line(img, start, end, Scalar.LimeGreen, 2);
                }

                Cv2.ImShow("test", img);
                Cv2.WaitKey(0);
            }
        }
    }
}
