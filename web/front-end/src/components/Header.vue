<template>
  <div id="Header">
    <div class="top-left-edition">
      <span style="color: #21b3b9; font-weight: bold">
        <span v-on:click="goFrontPage()" style="color: #21b3b9;text-decoration: underline">返回首页</span>
        <i class="el-icon-user" style="font-size: 23px"></i
        >,welcome
        <span id="name">{{this.$route.query.username}}</span>
      </span>
      <span>
        <i class="el-icon-phone-outline" style="font-size: 23px"></i>咨询电话：1234567890
      </span>
    </div>
    <div id="word">
      <h1>{{ msg }}</h1>
    </div>
  </div>
</template>
<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "基于深度学习的皮肤疾病识别系统",
      activeIndex: "1",
    };
  },
  methods: {
    goFrontPage(){
      this.$router.push({
        path: "/frontPage",
        name: "frontPage",
      })
    }
  },
};
</script>
<style scoped>
#Header {
  padding: 30px 90px 0 150px;
  width: 90%;
  margin: 10px auto;
}

#word {
  margin-left: 30%;
  margin-top: -35px;
  margin-bottom: 37px;
  height: 60px;
  line-height: 3.2em;
}

h1 {
  /*text-align: center;*/
  color: #21b3b9;
  letter-spacing: 5px;
  font-size: 1.8em;
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition span i {
  float: left;
  margin-right: 5px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

.top-left-edition span {
  float: left;
  font-size: 16px;
  color: #21b3b9;
  line-height: 24px;
  margin-right: 10px;
}

#name{
  margin-right: 3px;
}
</style>


