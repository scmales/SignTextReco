using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tesseract;

namespace SignTextReco
{
    class TextReco
    {
        public static IList<KeyValuePair<string, System.Drawing.Point>> GetTextPosition(Bitmap image, IList<System.Windows.Rect> bounds)
        {
            var resDic = new List<KeyValuePair<string, System.Drawing.Point>>();
            Mat srcImg = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);
            Mat binaryImg = new Mat();
            Cv2.Threshold(srcImg, binaryImg, 200, 255, ThresholdTypes.Binary);
            //
            foreach (var bound in bounds)
            {
                //scmales
                var x = bound.X;
                var y = bound.Y;
                var width = bound.Width;
                var height = bound.Height;
                if (bound.X < 0) x = 0;
                if (bound.Y < 0) y = 0;
                if (x + bound.Width > binaryImg.Width) width = binaryImg.Width - x;
                if (y + bound.Height > binaryImg.Height) height = binaryImg.Height - y;
                var tempRect = new OpenCvSharp.Rect(new OpenCvSharp.Point(x, y), new OpenCvSharp.Size(width, height));
                var tempMat = binaryImg[tempRect];
                Cv2.Threshold(tempMat, tempMat, -1, 255, ThresholdTypes.Binary);
            }
            var bitmapImg = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(binaryImg);
            try
            {
                //设置了中文包的路径在程序运行目录的/third-party/tessdata下
                using (var ocr = new TesseractEngine(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "third-party", "tessdata"), "chi_sim", EngineMode.Default))
                {
                    PageIteratorLevel myLevel = PageIteratorLevel.Word;
                    using (var img = PixConverter.ToPix(bitmapImg))
                    {
                        using (var page = ocr.Process(img))
                        {
                            using (var iter = page.GetIterator())
                            {
                                iter.Begin();
                                do
                                {
                                    if (iter.TryGetBoundingBox(myLevel, out var rect))
                                    {
                                        var curText = iter.GetText(myLevel);
                                        var curPoint = new System.Drawing.Point(rect.X1, rect.Y1 + rect.Height / 2);
                                        // frankro.
                                        if (string.IsNullOrWhiteSpace(curText))
                                            continue;

                                        resDic.Add(new KeyValuePair<string, System.Drawing.Point>(curText, curPoint));
                                    }
                                } while (iter.Next(myLevel));
                            }
                        }
                    }
                }
            }
            catch (Exception er)
            {
                //do nothing
                //MessageBox.Show(er.ToString());
            }
            return resDic;
        }
    }
}
