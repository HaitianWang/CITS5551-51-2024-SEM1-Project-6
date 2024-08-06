<template>
  <div id="Content">
    <el-dialog
      id="hello"
      title="怀疑您得了皮肤癌？我们的系统使用人工智能，对您的皮肤图片进行简单判断"
      :visible.sync="centerDialogVisible"
      width="70%"
      :before-close="handleClose"
    >

      <el-steps :active="3" align-center>
        <el-step title="步骤1" description="上传您的皮肤局部图片,例如某个斑点或痣"></el-step>
        <el-step title="步骤2" description="查看检测结果,会显示皮肤癌的区域与种类"></el-step>
        <el-step title="步骤3" description="查看皮肤癌科普和医院推荐"></el-step>
      </el-steps>
      <div style="margin-left: -400px;color: lightskyblue">上传图片示例：<div><img src="../../static/ISIC_7621283.jpg" height="55px" width="55px"></div></div>
    </el-dialog>

    <el-dialog
      title="AI预测中"
      :visible.sync="dialogTableVisible"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
    >
      <el-progress :percentage="percentage"></el-progress>
      <span slot="footer" class="dialog-footer">请耐心等待约3秒钟</span>
    </el-dialog>

    <div id="Kepu">
      <el-dialog
        id="kepudialog"
        title="皮肤癌科普"
        :visible.sync="kepuDialogVisible"
        width="70%"
        style="text-align: left"
      >
        <span style="color: blue">基底细胞癌</span>中国人常见的基底细胞癌类型为色素型和结节型。早期往往表现为头面部单发、缓慢发展的黑色斑块或者结节，部分中心可有破溃。也可以表现为皮肤颜色的斑块，或者红色的斑块，这种基底细胞癌的发病率非常高，根据北京大学的统计发现，基底细胞癌的患者数量占皮肤癌患者的三分之一以上。因此，基底细胞癌是非常常见的皮肤病，同时也是非常容易被误诊和漏诊的皮肤恶性肿瘤之一。
        <br/><span style="color: blue">黑色素瘤</span>是黑色素细胞恶变而来的肿瘤，恶性程度高，多发生于皮肤，也可发生在黏膜(包括内脏黏膜)等不同部位或组织，我国人群好发于肢端皮肤(足底、足趾、手指末端和甲下等部位)和 黏膜(鼻腔、口咽以及上、下消化道等)。我国皮肤黑色素瘤的发病率和死亡率相对较低，每30万人群中约有1例黑色素瘤患者,每50万癌症死亡人群中，约1例为黑色素瘤死亡。
        <br/><br/>全国治疗皮肤肿瘤医院排名比较靠前的有复旦大学附属华山医院、中国医学科学院皮肤病医院、北京大学第一医院等
        <br/><br/>置信区间给出的是被测量参数测量值的可信程度范围，可以通俗地理解成置信度越高是那种癌症的可能性越高，但是一切以医院检查为准
      </el-dialog>
    </div>

      <el-dialog
        title="AI预测中"
        :visible.sync="dialogTableVisible"
        :show-close="false"
        :close-on-press-escape="false"
        :append-to-body="true"
        :close-on-click-modal="false"
        :center="true"
      >
        <el-progress :percentage="percentage"></el-progress>
        <span slot="footer" class="dialog-footer">请耐心等待约3秒钟</span>
      </el-dialog>

    <div id="CT">
      <div id="CT_image">
        <span v-on:click="showJiaochen()" style="color:gray;text-decoration: underline;margin-left: -700px">还想看一眼使用教程？点这里！！</span>
        <el-card
          id="CT_image_1"
          class="box-card"
          style="
            border-radius: 8px;
            width: 800px;
            height: 360px;
            margin-bottom: -30px;
          "
        >
          <div class="demo-image__preview1">
            <div
              v-loading="loading"
              element-loading-text="上传图片中"
              element-loading-spinner="el-icon-loading"
            >
              <el-image
                :src="url_1"
                class="image_1"
                :preview-src-list="srcList"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">
                    <el-button
                      v-show="showbutton"
                      type="primary"
                      icon="el-icon-upload"
                      class="download_bt"
                      v-on:click="true_upload"
                    >
                      上传图像
                      <input
                        ref="upload"
                        style="display: none"
                        name="file"
                        type="file"
                        @change="update"
                      />
                    </el-button>
                  </div>
                </div>
              </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px">
              <span style="color: white">请上传您的皮肤图片(.jpg/.png)</span>
            </div>
          </div>
          <div class="demo-image__preview2">
            <div
              v-loading="loading"
              element-loading-text="处理中,请耐心等待"
              element-loading-spinner="el-icon-loading"
            >
              <el-image
                :src="url_2"
                class="image_1"
                :preview-src-list="srcList1"
                style="border-radius: 3px 3px 0 0"
              >
                <div slot="error">
                  <div slot="placeholder" class="error">{{ wait_return }}</div>
                </div>
              </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px">
              <span style="color: white; letter-spacing: 4px">检测结果</span>
            </div>
          </div>
        </el-card>
      </div>
      <div id="info_patient">
        <!-- 卡片放置表格 -->
        <el-card style="border-radius: 8px">
          <div slot="header" class="clearfix">
            <span>检测目标</span>
            <el-button
              style="margin-left: 35px"
              v-show="!showbutton"
              type="primary"
              icon="el-icon-upload"
              class="download_bt"
              v-on:click="true_upload2"
            >
              重新选择图像
              <input
                ref="upload2"
                style="display: none"
                name="file"
                type="file"
                @change="update"
              />
            </el-button>
          </div>
          <el-tabs v-model="activeName">
            <el-tab-pane label="检测到的目标" name="first" >
              <!-- 表格存放特征值 -->
              <span><span v-on:click="showKepu()" style="color:blue;text-decoration: underline;margin-left: 0px">点这里</span>查看癌症科普。如果没有检测结果，就是没有检测到癌症。本系统不负任何法律责任，请去医院确认</span>
              <el-table
                :data="feature_list"
                height="250"
                border
                style="width: 750px; text-align: center"
                v-loading="loading"
                element-loading-text="数据正在处理中，请耐心等待"
                element-loading-spinner="el-icon-loading"
                lazy
              >
                <el-table-column label="目标类别" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[2] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="目标大小" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[0] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="置信度" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[1] }}</span>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "Content",
  data() {
    return {
      server_url: "http://127.0.0.1:5003",
      activeName: "first",
      active: 0,
      centerDialogVisible: true,
      kepuDialogVisible: false,
      url_1: "",
      url_2: "",
      textarea: "",
      srcList: [],
      srcList1: [],
      feature_list: [],
      feature_list_1: [],
      feat_list: [],
      url: "",
      visible: false,
      wait_return: "等待上传",
      wait_upload: "等待上传",
      loading: false,
      table: false,
      isNav: false,
      showbutton: true,
      percentage: 0,
      fullscreenLoading: false,
      opacitys: {
        opacity: 0,
      },
      dialogTableVisible: false,
    };
  },
  created: function () {
    document.title = "YOLOv5目标检测WEB端";
  },
  methods: {
    true_upload() {
      this.$refs.upload.click();
    },
    true_upload2() {
      this.$refs.upload2.click();
    },
    showJiaochen(){
      this.centerDialogVisible=true;
    },
    showKepu(){
      this.kepuDialogVisible=true;
    },
    handleClose(done) {
      this.$confirm("确认关闭？")
        .then((_) => {
          done();
        })
        .catch((_) => {});
    },
    next() {
      this.active++;
    },
    // 获得目标文件
    getObjectURL(file) {
      var url = null;
      if (window.createObjcectURL != undefined) {
        url = window.createOjcectURL(file);
      } else if (window.URL != undefined) {
        url = window.URL.createObjectURL(file);
      } else if (window.webkitURL != undefined) {
        url = window.webkitURL.createObjectURL(file);
      }
      return url;
    },
    // 上传文件
    update(e) {
      this.percentage = 0;
      this.dialogTableVisible = true;
      this.url_1 = "";
      this.url_2 = "";
      this.srcList = [];
      this.srcList1 = [];
      this.wait_return = "";
      this.wait_upload = "";
      this.feature_list = [];
      this.feat_list = [];
      this.fullscreenLoading = true;
      this.loading = true;
      this.showbutton = false;
      let file = e.target.files[0];
      this.url_1 = this.$options.methods.getObjectURL(file);
      let param = new FormData(); //创建form对象
      param.append("file", file, file.name); //通过append向form对象添加数据
      var timer = setInterval(() => {
        this.myFunc();
      }, 30);
      let config = {
        headers: { "Content-Type": "multipart/form-data" },
      }; //添加请求头
      axios
        .post(this.server_url + "/upload", param, config)
        .then((response) => {
          this.percentage = 100;
          clearInterval(timer);
          this.url_1 = response.data.image_url;
          this.srcList.push(this.url_1);
          this.url_2 = response.data.draw_url;
          this.srcList1.push(this.url_2);
          this.fullscreenLoading = false;
          this.loading = false;

          this.feat_list = Object.keys(response.data.image_info);

          for (var i = 0; i < this.feat_list.length; i++) {
            response.data.image_info[this.feat_list[i]][2] = this.feat_list[i];
            this.feature_list.push(response.data.image_info[this.feat_list[i]]);
          }

          this.feature_list.push(response.data.image_info);

          //console.log("list:"+this.feature_list[0][2])
          for (var i = 0; i < this.feature_list.length-1; i++) {
            console.log("len:"+this.feature_list.length)
            console.log("list:"+this.feature_list[i][2])
            if (this.feature_list[i][2].includes("BasalCellCarcinoma")) {
              this.feature_list[i][2] = this.feature_list[i][2] + "基底细胞癌"
            }
            else if(this.feature_list[i][2].includes("Melanoma")){
              this.feature_list[i][2] = this.feature_list[i][2] + "黑色素瘤"
            }
          }

          this.feature_list_1 = this.feature_list[0];
          this.dialogTableVisible = false;
          this.percentage = 0;
          this.notice1();
        });
    },
    myFunc() {
      if (this.percentage + 33 < 99) {
        this.percentage = this.percentage + 33;
      } else {
        this.percentage = 99;
      }
    },
    drawChart() {},
    notice1() {
      this.$notify({
        title: "预测成功",
        message: "点击图片可以查看大图",
        duration: 0,
        type: "success",
      });
    },
    notice2() {
      this.$notify({
        title: "怀疑自己得了皮肤癌吗？",
        message: "",
        duration: 0,
        type: "success",
      });
    },
  },
  mounted() {
    this.drawChart();
  },
};
</script>

<style>
.el-button {
  padding: 12px 20px !important;
}

#hello p {
  font-size: 15px !important;
  /*line-height: 25px;*/
}

.n1 .el-step__description {
  padding-right: 20%;
  font-size: 14px;
  line-height: 20px;
  /* font-weight: 400; */
}
</style>

<style scoped>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.dialog_info {
  margin: 20px auto;
}

.text {
  font-size: 14px;
}

.item {
  margin-bottom: 18px;
}

.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}

.clearfix:after {
  clear: both;
}

.box-card {
  width: 680px;
  height: 200px;
  border-radius: 8px;
  margin-top: -20px;
}

.divider {
  width: 50%;
}

#CT {
  display: flex;
  height: 100%;
  width: 100%;
  flex-wrap: wrap;
  justify-content: center;
  margin: 0 auto;
  margin-right: 0px;
  max-width: 1800px;
}

#CT_image_1 {
  width: 90%;
  height: 40%;
  margin: 0px auto;
  padding: 0px auto;
  margin-right: 180px;
  margin-bottom: 0px;
  border-radius: 4px;
}

#CT_image {
  margin-bottom: 60px;
  margin-left: 30px;
  margin-top: 5px;
}

.image_1 {
  width: 275px;
  height: 260px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.img_info_1 {
  height: 30px;
  width: 275px;
  text-align: center;
  background-color: #21b3b9;
  line-height: 30px;
}

.demo-image__preview1 {
  width: 250px;
  height: 290px;
  margin: 20px 60px;
  float: left;
}

.demo-image__preview2 {
  width: 250px;
  height: 290px;

  margin: 20px 460px;
  /* background-color: green; */
}

.error {
  margin: 100px auto;
  width: 50%;
  padding: 10px;
  text-align: center;
}

.block-sidebar {
  position: fixed;
  display: none;
  left: 50%;
  margin-left: 600px;
  top: 350px;
  width: 60px;
  z-index: 99;
}

.block-sidebar .block-sidebar-item {
  font-size: 50px;
  color: lightblue;
  text-align: center;
  line-height: 50px;
  margin-bottom: 20px;
  cursor: pointer;
  display: block;
}

div {
  display: block;
}

.block-sidebar .block-sidebar-item:hover {
  color: #187aab;
}

.download_bt {
  padding: 10px 16px !important;
}

#upfile {
  width: 104px;
  height: 45px;
  background-color: #187aab;
  color: #fff;
  text-align: center;
  line-height: 45px;
  border-radius: 3px;
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
  color: #fff;
  font-family: "Source Sans Pro", Verdana, sans-serif;
  font-size: 0.875rem;
}

.file {
  width: 200px;
  height: 130px;
  position: absolute;
  left: -20px;
  top: 0;
  z-index: 1;
  -moz-opacity: 0;
  -ms-opacity: 0;
  -webkit-opacity: 0;
  opacity: 0; /*css属性&mdash;&mdash;opcity不透明度，取值0-1*/
  filter: alpha(opacity=0);
  cursor: pointer;
}

#upload {
  position: relative;
  margin: 0px 0px;
}

#Content {
  width: 85%;
  height: 800px;
  background-color: #ffffff;
  margin: 15px auto;
  display: flex;
  min-width: 1200px;
}

.divider {
  background-color: #eaeaea !important;
  height: 2px !important;
  width: 100%;
  margin-bottom: 50px;
}

.divider_1 {
  background-color: #ffffff;
  height: 2px !important;
  width: 100%;
  margin-bottom: 20px;
  margin: 20px auto;
}

.steps {
  font-family: "lucida grande", "lucida sans unicode", lucida, helvetica,
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
  color: #21b3b9;
  text-align: center;
  margin: 15px auto;
  font-size: 20px;
  font-weight: bold;
  text-align: center;
}

.step_1 {
  /*color: #303133 !important;*/
  margin: 20px 26px;
}

#info_patient {
  margin-top: 10px;
  margin-right: 160px;
}
</style>


