<template>
  <div id="Content">

    <el-dialog
      title="AI Prediction in Progress"
      :visible.sync="dialogTableVisible"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
    >
      <el-progress :percentage="percentage"></el-progress>
      <span slot="footer" class="dialog-footer">Please wait patiently for about 3 seconds</span>
    </el-dialog>

    <div id="CT">
      <div id="CT_image">
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
              element-loading-text="Uploading Image"
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
                      Upload Image
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
              <span style="color: white">Please upload your farmland image</span>
            </div>
          </div>
          <div class="demo-image__preview2">
            <div
              v-loading="loading"
              element-loading-text="Processing, please wait"
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
              <span style="color: white; letter-spacing: 4px">Detection Result</span>
            </div>
          </div>
        </el-card>
      </div>
      <div id="info_patient">
        <!-- 卡片放置表格 -->
        <el-card style="border-radius: 8px">
          <div slot="header" class="clearfix">
            <span>Detection Target</span>
            <el-button
              style="margin-left: 35px"
              v-show="!showbutton"
              type="primary"
              icon="el-icon-upload"
              class="download_bt"
              v-on:click="true_upload2"
            >
              Re-select Image
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
            <el-tab-pane label="Detected Targets" name="first" >
              <!-- 表格存放特征值 -->
              <el-table
                :data="feature_list"
                height="250"
                border
                style="width: 750px; text-align: center"
                v-loading="loading"
                element-loading-text="Data is being processed, please wait"
                element-loading-spinner="el-icon-loading"
                lazy
              >
                <el-table-column label="Target Category" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[2] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="Target Size" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[0] }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="Confidence" width="250px">
                  <template slot-scope="scope">
                    <span>{{ scope.row[1] }}</span>
                  </template>
                </el-table-column>
                <!-- 添加空数据时显示 "No data" -->
                <template slot="empty">
                  <div class="no-data">No data</div>
                </template>
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
        wait_return: "Waiting for upload",
        wait_upload: "Waiting for upload",
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
      document.title = "UWAIntelliCrop";
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
        this.$confirm("Confirm to close?")
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
        let param = new FormData(); // 创建 form 对象
        param.append("file", file, file.name); // 通过 append 向 form 对象添加数据
        var timer = setInterval(() => {
          this.myFunc();
        }, 30);
        let config = {
          headers: { "Content-Type": "multipart/form-data" },
        }; // 添加请求头
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
                this.feature_list[i][2] = this.feature_list[i][2] + " Basal Cell Carcinoma"
              }
              else if(this.feature_list[i][2].includes("Melanoma")){
                this.feature_list[i][2] = this.feature_list[i][2] + " Melanoma"
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
          title: "Prediction Successful",
          message: "Click the image to view the large image",
          duration: 0,
          type: "success",
        });
      },
      notice2() {
        this.$notify({
          title: "Suspect you have skin cancer?",
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
  }

  .n1 .el-step__description {
    padding-right: 20%;
    font-size: 14px;
    line-height: 20px;
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

  .block-sidebar .block-sidebar-item:hover {
    color: #187aab;
  }

  .download_bt {
    padding: 10px 16px !important;
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
    margin: 20px 26px;
  }

  #info_patient {
    margin-top: 10px;
    margin-right: 160px;
  }

  .no-data {
    color: #c0c4cc;
    font-size: 14px;
    text-align: center;
    line-height: 250px; /* Adjust this value to vertically center the text */
  }
</style>
